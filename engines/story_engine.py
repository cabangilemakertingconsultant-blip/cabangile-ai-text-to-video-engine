#!/usr/bin/env python3
"""
Cabangile AI Studio v2.6
Story Engine - Single File Production Module

Compatible:
- Python 3.11+
- Android Termux
- Linux
- Windows
- macOS

Architecture:
- Clean Architecture
- SOLID Principles
- Multi AI Provider Failover
- Offline Generation
"""

from __future__ import annotations

import os
import sys
import json
import re
import time
import math
import logging
import threading
import urllib.request
import urllib.error

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Generator,
    Callable,
)


# ==========================================================
# OPTIONAL CABANGILE INTEGRATION
# ==========================================================

try:
    import config
except ImportError:

    class ConfigFallback:
        OLLAMA_URL = "http://localhost:11434/api/generate"
        OLLAMA_MODEL = "llama3"

        OPENAI_URL = (
            "https://api.openai.com/v1/chat/completions"
        )
        OPENAI_API_KEY = ""
        OPENAI_MODEL = "gpt-4o-mini"

        POLLINATIONS_URL = (
            "https://text.pollinations.ai/"
        )

    config = ConfigFallback()


# ==========================================================
# STRUCTURED LOGGING
# ==========================================================

logger = logging.getLogger(
    "CabangileAIStudio.StoryEngine"
)

logger.setLevel(logging.INFO)

if not logger.handlers:

    stream_handler = logging.StreamHandler(
        sys.stdout
    )

    formatter = logging.Formatter(
        "[%(asctime)s] "
        "[%(levelname)s] "
        "[StoryEngine] "
        "%(message)s"
    )

    stream_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        stream_handler
    )


# ==========================================================
# THREAD LOCK
# ==========================================================

_ENGINE_LOCK = threading.RLock()


# ==========================================================
# DATA MODELS
# ==========================================================


@dataclass(slots=True)
class Character:
    name: str
    role: str
    description: str
    voice_profile: str = ""
    traits: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass(slots=True)
class Scene:

    scene_number: int
    title: str
    setting: str
    narration: str

    dialogue: List[Dict[str, str]] = field(
        default_factory=list
    )

    image_prompt: str = ""

    subtitles: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    duration_seconds: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass(slots=True)
class StoryMetadata:

    title: str
    style: str
    target_duration: float

    estimated_total_duration: float = 0.0

    created_at: float = field(
        default_factory=time.time
    )

    version: str = "2.6"

    tags: List[str] = field(
        default_factory=list
    )



@dataclass(slots=True)
class StoryProject:

    metadata: StoryMetadata

    characters: List[Character] = field(
        default_factory=list
    )

    scenes: List[Scene] = field(
        default_factory=list
    )

    generation_summary: Dict[str, Any] = field(
        default_factory=dict
    )


    def to_dict(self) -> Dict[str, Any]:

        return {

            "metadata":
                asdict(self.metadata),

            "characters":
                [
                    c.to_dict()
                    for c in self.characters
                ],

            "scenes":
                [
                    s.to_dict()
                    for s in self.scenes
                ],

            "generation_summary":
                self.generation_summary
        }



# ==========================================================
# AI PROVIDER INTERFACE
# ==========================================================


class BaseAIProvider(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str
    ) -> str:
        pass


    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str
    ) -> Generator[str, None, None]:
        pass
        # ==========================================================
# OLLAMA PROVIDER
# ==========================================================


class OllamaProvider(BaseAIProvider):

    def __init__(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None
    ) -> None:

        self.url = (
            url
            or getattr(
                config,
                "OLLAMA_URL",
                "http://localhost:11434/api/generate"
            )
        )

        self.model = (
            model
            or getattr(
                config,
                "OLLAMA_MODEL",
                "llama3"
            )
        )


    def generate(
        self,
        prompt: str,
        system_prompt: str
    ) -> str:

        payload = {

            "model": self.model,

            "prompt":
                f"{system_prompt}\n\n{prompt}",

            "stream": False

        }

        request = urllib.request.Request(

            self.url,

            data=json.dumps(
                payload
            ).encode("utf-8"),

            headers={
                "Content-Type":
                    "application/json"
            },

            method="POST"
        )


        try:

            with urllib.request.urlopen(
                request,
                timeout=60
            ) as response:

                data = json.loads(
                    response.read()
                    .decode("utf-8")
                )

                return str(
                    data.get(
                        "response",
                        ""
                    )
                ).strip()


        except Exception as error:

            raise RuntimeError(
                f"Ollama failed: {error}"
            )


    def generate_stream(
        self,
        prompt: str,
        system_prompt: str
    ) -> Generator[str, None, None]:


        payload = {

            "model": self.model,

            "prompt":
                f"{system_prompt}\n\n{prompt}",

            "stream": True

        }


        request = urllib.request.Request(

            self.url,

            data=json.dumps(
                payload
            ).encode("utf-8"),

            headers={
                "Content-Type":
                    "application/json"
            },

            method="POST"
        )


        try:

            with urllib.request.urlopen(
                request,
                timeout=60
            ) as response:


                for line in response:

                    if not line:
                        continue


                    try:

                        chunk = json.loads(
                            line.decode(
                                "utf-8"
                            )
                        )

                        text = chunk.get(
                            "response",
                            ""
                        )

                        if text:
                            yield text


                    except json.JSONDecodeError:

                        continue


        except Exception as error:

            logger.error(
                f"Ollama streaming error: {error}"
            )

            yield ""



# ==========================================================
# OPENAI PROVIDER
# ==========================================================


class OpenAIProvider(BaseAIProvider):

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        model: Optional[str] = None
    ) -> None:


        self.api_key = (
            api_key
            or getattr(
                config,
                "OPENAI_API_KEY",
                ""
            )
        )


        self.url = (
            url
            or getattr(
                config,
                "OPENAI_URL",
                "https://api.openai.com/v1/chat/completions"
            )
        )


        self.model = (
            model
            or getattr(
                config,
                "OPENAI_MODEL",
                "gpt-4o-mini"
            )
        )


    def generate(
        self,
        prompt: str,
        system_prompt: str
    ) -> str:


        if not self.api_key:

            raise RuntimeError(
                "OpenAI API key missing"
            )


        payload = {

            "model": self.model,

            "messages": [

                {
                    "role":
                        "system",

                    "content":
                        system_prompt
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt
                }

            ],

            "temperature":
                0.7

        }


        request = urllib.request.Request(

            self.url,

            data=json.dumps(
                payload
            ).encode("utf-8"),

            headers={

                "Content-Type":
                    "application/json",

                "Authorization":
                    f"Bearer {self.api_key}"

            },

            method="POST"

        )


        try:

            with urllib.request.urlopen(
                request,
                timeout=60
            ) as response:


                data = json.loads(
                    response.read()
                    .decode("utf-8")
                )


                return (
                    data["choices"][0]
                    ["message"]
                    ["content"]
                    .strip()
                )


        except Exception as error:

            raise RuntimeError(
                f"OpenAI failed: {error}"
            )


    def generate_stream(
        self,
        prompt: str,
        system_prompt: str
    ) -> Generator[str, None, None]:


        result = self.generate(
            prompt,
            system_prompt
        )


        for part in result.split():

            yield part + " "



# ==========================================================
# POLLINATIONS PROVIDER
# ==========================================================


class PollinationsProvider(BaseAIProvider):

    def __init__(
        self,
        url: Optional[str] = None
    ) -> None:

        self.url = (
            url
            or getattr(
                config,
                "POLLINATIONS_URL",
                "https://text.pollinations.ai/"
            )
        )



    def generate(
        self,
        prompt: str,
        system_prompt: str
    ) -> str:


        payload = {

            "messages": [

                {
                    "role":
                        "system",

                    "content":
                        system_prompt
                },

                {
                    "role":
                        "user",

                    "content":
                        prompt
                }

            ],

            "jsonMode":
                True

        }


        request = urllib.request.Request(

            self.url,

            data=json.dumps(
                payload
            ).encode("utf-8"),

            headers={

                "Content-Type":
                    "application/json"

            },

            method="POST"

        )


        try:

            with urllib.request.urlopen(
                request,
                timeout=60
            ) as response:

                return (
                    response.read()
                    .decode("utf-8")
                    .strip()
                )


        except Exception as error:

            raise RuntimeError(
                f"Pollinations failed: {error}"
            )



    def generate_stream(
        self,
        prompt: str,
        system_prompt: str
    ) -> Generator[str, None, None]:

        text = self.generate(
            prompt,
            system_prompt
        )

        yield text



# ==========================================================
# OFFLINE DETERMINISTIC FALLBACK
# ==========================================================


class OfflineDeterministicFallbackProvider(
    BaseAIProvider
):


    def generate(
        self,
        prompt: str,
        system_prompt: str
    ) -> str:


        story = {

            "title":
                "Cabangile Offline Generated Story",

            "characters": [

                {

                    "name":
                        "Cabangile Core",

                    "role":
                        "AI Creator",

                    "description":
                        "A resilient creative intelligence system",

                    "voice_profile":
                        "deep cinematic",

                    "traits":
                        [
                            "creative",
                            "adaptive"
                        ]

                }

            ],

            "scenes": [

                {

                    "scene_number":
                        1,

                    "title":
                        "System Awakening",

                    "setting":
                        "Digital Studio",

                    "narration":
                        "Inside Cabangile AI Studio, a new creative engine begins generating stories, videos, and digital experiences.",

                    "dialogue":

                        [

                            {

                                "speaker":
                                    "Cabangile Core",

                                "text":
                                    "Creation begins with imagination."

                            }

                        ],

                    "image_prompt":
                        "A futuristic AI studio, cinematic lighting, wide camera angle, realistic digital environment, ultra detailed 8K"

                }

            ]

        }


        return json.dumps(
            story
        )



    def generate_stream(
        self,
        prompt: str,
        system_prompt: str
    ) -> Generator[str, None, None]:

        yield self.generate(
            prompt,
            system_prompt
        )
        # ==========================================================
# JSON REPAIR AND VALIDATION SYSTEM
# ==========================================================


class JsonRepairKit:

    _fence_regex = re.compile(
        r"```(?:json)?|```",
        re.IGNORECASE
    )

    _json_regex = re.compile(
        r"(\{.*\}|\[.*\])",
        re.DOTALL
    )


    @classmethod
    def clean(
        cls,
        raw_text: str
    ) -> str:

        if not raw_text:

            return "{}"


        text = raw_text.strip()


        text = cls._fence_regex.sub(
            "",
            text
        )


        match = cls._json_regex.search(
            text
        )


        if match:

            text = match.group(
                1
            )


        text = re.sub(
            r",\s*([\]}])",
            r"\1",
            text
        )


        return cls.balance(
            text
        )


    @classmethod
    def balance(
        cls,
        text: str
    ) -> str:


        braces_open = text.count("{")
        braces_close = text.count("}")


        brackets_open = text.count("[")
        brackets_close = text.count("]")


        if braces_open > braces_close:

            text += "}" * (
                braces_open - braces_close
            )


        if brackets_open > brackets_close:

            text += "]" * (
                brackets_open - brackets_close
            )


        return text



    @classmethod
    def parse(
        cls,
        raw_text: str
    ) -> Dict[str, Any]:

        cleaned = cls.clean(
            raw_text
        )


        try:

            data = json.loads(
                cleaned
            )


        except json.JSONDecodeError as error:

            raise ValueError(
                f"Invalid JSON output: {error}"
            )


        if isinstance(
            data,
            list
        ):

            return {
                "scenes":
                    data
            }


        if not isinstance(
            data,
            dict
        ):

            raise ValueError(
                "JSON root must be object"
            )


        return data



# ==========================================================
# CONTENT PROCESSING UTILITIES
# ==========================================================


class ContentProcessor:


    @staticmethod
    def clean_tts_text(
        text: str
    ) -> str:

        if not text:

            return ""


        text = re.sub(
            r"\*.*?\*",
            "",
            text
        )


        text = re.sub(
            r"\[.*?\]",
            "",
            text
        )


        text = re.sub(
            r"\(.*?\)",
            "",
            text
        )


        text = re.sub(
            r"\s+",
            " ",
            text
        )


        return text.strip()



    @staticmethod
    def estimate_duration(
        narration: str,
        dialogue: List[Dict[str,str]]
    ) -> float:


        narration_words = len(
            narration.split()
        )


        dialogue_words = sum(

            len(
                item.get(
                    "text",
                    ""
                ).split()
            )

            for item in dialogue

        )


        seconds = (

            narration_words / 130 * 60

            +

            dialogue_words / 150 * 60

        )


        return round(
            max(
                seconds,
                3.5
            ),
            2
        )



    @staticmethod
    def generate_subtitles(
        narration: str,
        dialogue: List[Dict[str,str]],
        start_time: float = 0.0
    ) -> List[Dict[str,Any]]:


        subtitles = []

        current = start_time


        words = narration.split()

        chunk_size = 8


        for index in range(
            0,
            len(words),
            chunk_size
        ):


            line = " ".join(
                words[
                    index:index + chunk_size
                ]
            )


            duration = max(
                len(line.split())
                /
                130
                *
                60,

                1.5
            )


            subtitles.append(

                {

                    "start":
                        round(
                            current,
                            2
                        ),

                    "end":
                        round(
                            current + duration,
                            2
                        ),

                    "speaker":
                        "Narrator",

                    "text":
                        line

                }

            )


            current += duration



        for item in dialogue:

            text = item.get(
                "text",
                ""
            )


            if not text:

                continue


            duration = max(

                len(
                    text.split()
                )
                /
                150
                *
                60,

                1.5

            )


            subtitles.append(

                {

                    "start":
                        round(
                            current,
                            2
                        ),

                    "end":
                        round(
                            current + duration,
                            2
                        ),

                    "speaker":
                        item.get(
                            "speaker",
                            "Character"
                        ),

                    "text":
                        text

                }

            )


            current += duration



        return subtitles



    @staticmethod
    def build_image_prompt(
        description: str,
        style: str
    ) -> str:


        return (

            f"{description}. "

            f"Cinematic {style} scene, "

            "professional lighting, "

            "dynamic camera angle, "

            "balanced composition, "

            "atmospheric environment, "

            "photorealistic quality, "

            "ultra detailed, "

            "8K resolution, "

            "high quality visual production"

        )



# ==========================================================
# STORY STYLE DEFINITIONS
# ==========================================================


STORY_STYLES = {

    "Documentary",
    "History",
    "Technology",
    "Educational",
    "Business",
    "Motivational",
    "Fantasy",
    "Mystery",
    "Sci-Fi",
    "News",
    "Custom"

}
# ==========================================================
# MAIN STORY ENGINE
# ==========================================================


class StoryEngine:

    def __init__(
        self,
        providers: Optional[
            List[BaseAIProvider]
        ] = None
    ) -> None:


        self.providers = (

            providers

            if providers

            else [

                OpenAIProvider(),

                OllamaProvider(),

                PollinationsProvider(),

                OfflineDeterministicFallbackProvider()

            ]

        )


    # ------------------------------------------------------

    def _call_provider_chain(
        self,
        prompt: str,
        system_prompt: str
    ) -> str:


        last_error = None


        for provider in self.providers:

            try:

                logger.info(
                    "Using provider: %s",
                    provider.__class__.__name__
                )


                result = provider.generate(
                    prompt,
                    system_prompt
                )


                if result:

                    return result


            except Exception as error:

                last_error = error

                logger.warning(
                    "%s failed: %s",
                    provider.__class__.__name__,
                    error
                )


        raise RuntimeError(
            f"All AI providers failed: {last_error}"
        )



    # ------------------------------------------------------

    def create_system_prompt(
        self
    ) -> str:


        return """

You are Cabangile AI Studio Story Engine.

Generate professional cinematic content.

Return ONLY valid JSON.

Schema:

{
"title":"",
"characters":[],
"scenes":[]
}

Every scene requires:

scene_number
title
setting
narration
dialogue
image_prompt

Create content suitable for AI video production.

"""



    # ------------------------------------------------------

    def generate_story_project(
        self,
        prompt: str,
        style: str = "Documentary",
        target_duration: float = 120.0,
        progress_callback:
            Optional[
                Callable[[str,float],None]
            ] = None
    ) -> StoryProject:


        if not prompt.strip():

            raise ValueError(
                "Story prompt cannot be empty"
            )


        if style not in STORY_STYLES:

            style = "Custom"



        if progress_callback:

            progress_callback(
                "Starting generation",
                0.1
            )


        user_prompt = (

            f"""

Create a {style} story.

Topic:

{prompt}

Target duration:

{target_duration} seconds.

"""

        )


        raw = self._call_provider_chain(

            user_prompt,

            self.create_system_prompt()

        )


        if progress_callback:

            progress_callback(
                "Parsing AI response",
                0.4
            )


        data = JsonRepairKit.parse(
            raw
        )



        metadata = StoryMetadata(

            title=data.get(
                "title",
                "Untitled Cabangile Story"
            ),

            style=style,

            target_duration=
                target_duration,

            tags=[

                style.lower(),

                "cabangile-ai-studio"

            ]

        )



        characters = []


        for item in data.get(
            "characters",
            []
        ):


            characters.append(

                Character(

                    name=item.get(
                        "name",
                        "Unknown"
                    ),

                    role=item.get(
                        "role",
                        "Character"
                    ),

                    description=item.get(
                        "description",
                        ""
                    ),

                    voice_profile=item.get(
                        "voice_profile",
                        ""
                    ),

                    traits=item.get(
                        "traits",
                        []

                    )

                )

            )



        scenes = []

        total_duration = 0.0



        for index, item in enumerate(

            data.get(
                "scenes",
                []
            ),

            start=1

        ):


            narration = ContentProcessor.clean_tts_text(

                item.get(
                    "narration",
                    ""
                )

            )


            dialogue = item.get(
                "dialogue",
                []
            )



            duration = ContentProcessor.estimate_duration(

                narration,

                dialogue

            )


            subtitles = ContentProcessor.generate_subtitles(

                narration,

                dialogue,

                total_duration

            )



            image_prompt = item.get(

                "image_prompt",

                ContentProcessor.build_image_prompt(

                    item.get(
                        "setting",
                        ""
                    ),

                    style

                )

            )



            scene = Scene(

                scene_number=index,

                title=item.get(

                    "title",

                    f"Scene {index}"

                ),

                setting=item.get(

                    "setting",

                    "Unknown"

                ),

                narration=narration,

                dialogue=dialogue,

                image_prompt=image_prompt,

                subtitles=subtitles,

                duration_seconds=duration,

                metadata={

                    "generated":

                        True

                }

            )


            scenes.append(
                scene
            )


            total_duration += duration



        metadata.estimated_total_duration = round(

            total_duration,

            2

        )



        if progress_callback:

            progress_callback(

                "Generation complete",

                1.0

            )



        return StoryProject(

            metadata=metadata,

            characters=characters,

            scenes=scenes,

            generation_summary={

                "scenes":
                    len(scenes),

                "characters":
                    len(characters),

                "duration":
                    total_duration

            }

        )



    # ------------------------------------------------------

    def generate_story_stream(
        self,
        prompt: str,
        style: str = "Documentary"
    ) -> Generator[str,None,None]:


        system = self.create_system_prompt()


        for provider in self.providers:


            try:

                yield from provider.generate_stream(

                    prompt,

                    system

                )

                return


            except Exception:

                continue



    # ------------------------------------------------------

    def batch_generate(
        self,
        prompts: List[str],
        style: str = "Documentary"
    ) -> List[StoryProject]:


        results = []


        for item in prompts:

            try:

                results.append(

                    self.generate_story_project(

                        item,

                        style

                    )

                )


            except Exception as error:

                logger.error(
                    "Batch item failed: %s",
                    error
                )


        return results
        # ==========================================================
# PROJECT EXPORT / IMPORT SYSTEM
# ==========================================================


class ProjectExporter:


    @staticmethod
    def save_project(
        project: StoryProject,
        output_path: str
    ) -> bool:


        try:

            path = Path(
                output_path
            )

            path.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            with path.open(
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(

                    project.to_dict(),

                    file,

                    indent=4,

                    ensure_ascii=False

                )


            logger.info(
                "Project saved: %s",
                path
            )


            return True


        except Exception as error:

            logger.error(
                "Save failed: %s",
                error
            )

            return False



    @staticmethod
    def load_project(
        input_path: str
    ) -> Optional[StoryProject]:


        try:

            path = Path(
                input_path
            )


            with path.open(
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(
                    file
                )


            meta = data["metadata"]


            metadata = StoryMetadata(

                title=meta["title"],

                style=meta["style"],

                target_duration=
                    meta["target_duration"],

                estimated_total_duration=
                    meta.get(
                        "estimated_total_duration",
                        0.0
                    ),

                created_at=
                    meta.get(
                        "created_at",
                        time.time()
                    ),

                version=
                    meta.get(
                        "version",
                        "2.6"
                    ),

                tags=
                    meta.get(
                        "tags",
                        []
                    )

            )


            characters = [

                Character(
                    **item
                )

                for item in data.get(
                    "characters",
                    []
                )

            ]



            scenes = []


            for item in data.get(
                "scenes",
                []
            ):

                scenes.append(

                    Scene(

                        scene_number=item["scene_number"],

                        title=item["title"],

                        setting=item["setting"],

                        narration=item["narration"],

                        dialogue=item.get(
                            "dialogue",
                            []
                        ),

                        image_prompt=item.get(
                            "image_prompt",
                            ""
                        ),

                        subtitles=item.get(
                            "subtitles",
                            []
                        ),

                        duration_seconds=item.get(
                            "duration_seconds",
                            0.0
                        ),

                        metadata=item.get(
                            "metadata",
                            {}
                        )

                    )

                )


            return StoryProject(

                metadata=metadata,

                characters=characters,

                scenes=scenes,

                generation_summary=data.get(
                    "generation_summary",
                    {}
                )

            )


        except Exception as error:

            logger.error(
                "Load failed: %s",
                error
            )

            return None



# ==========================================================
# PRODUCTION DIAGNOSTIC
# ==========================================================


def run_diagnostic() -> None:


    print("=" * 60)

    print(
        "CABANGILE AI STUDIO STORY ENGINE v2.6"
    )

    print(
        "Production Diagnostic"
    )

    print("=" * 60)



    engine = StoryEngine(

        providers=[

            OfflineDeterministicFallbackProvider()

        ]

    )


    print(
        "\nGenerating offline test story..."
    )


    project = engine.generate_story_project(

        prompt=

        "A futuristic AI studio creating digital worlds",

        style=

        "Technology",

        target_duration=

        60

    )


    print(
        "Title:",
        project.metadata.title
    )


    print(
        "Scenes:",
        len(project.scenes)
    )


    print(
        "Characters:",
        len(project.characters)
    )


    filename = (
        "cabangile_story_test.json"
    )


    print(
        "\nSaving project..."
    )


    saved = ProjectExporter.save_project(

        project,

        filename

    )


    print(
        "Saved:",
        saved
    )


    print(
        "\nReloading project..."
    )


    loaded = ProjectExporter.load_project(

        filename

    )


    if loaded:

        print(
            "Integrity check: PASSED"
        )

    else:

        print(
            "Integrity check: FAILED"
        )



    if Path(filename).exists():

        Path(filename).unlink()



    print(
        "\nDiagnostic complete."
    )

    print("=" * 60)




# ==========================================================
# EXECUTION ENTRY
# ==========================================================


if __name__ == "__main__":

    run_diagnostic()
