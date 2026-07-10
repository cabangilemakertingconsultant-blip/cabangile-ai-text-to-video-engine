/**
 * ============================================================
 * Cabangile AI Studio
 * AI Engine
 * Central AI Orchestrator
 * Version 1.0.0
 * ============================================================
 */

import eventBus from "./event-bus.js";

import TaskQueue from "./ai/task-queue.js";
import ProviderManager from "./ai/provider-manager.js";

import StoryEngine from "./ai/story-engine.js";
import ScreenplayEngine from "./ai/screenplay-engine.js";
import SceneEngine from "./ai/scene-engine.js";
import CharacterEngine from "./ai/character-engine.js";
import DialogueEngine from "./ai/dialogue-engine.js";
import PromptEngine from "./ai/prompt-engine.js";

export default class AIEngine {
    async generateMovieProject(prompt) {

        if (!prompt || prompt.trim() === "") {
            throw new Error("Prompt cannot be empty.");
        }

        const project = await this.generateStory(prompt);

        await this.generateScenes(project);

        await this.generateCharacters(project);

        await this.generateDialogue(project);

        project.imagePrompts = project.scenes.map(scene =>
            this.prompt.createImagePrompt(scene)
        );

        project.videoPrompts = project.scenes.map(scene =>
            this.prompt.createVideoPrompt(scene)
        );

        eventBus.emit("project:complete", project);

        return project;
    }

    reset() {

        this.queue.clear();

        eventBus.emit("ai:reset");

    }
    constructor() {
    async generateStory(prompt) {

        if (!this.initialized) {
            await this.initialize();
        }

        eventBus.emit("story:start", { prompt });

        const story = await this.story.generate(prompt);

        const screenplay = this.screenplay.generate(story);

        story.screenplay = screenplay;

        eventBus.emit("story:complete", story);

        return story;

    }

    async generateScenes(story) {

        const scenes = [];

        for (const screenplayScene of story.screenplay.scenes) {

            scenes.push(
                this.scene.create(screenplayScene)
            );

        }

        story.scenes = scenes;

        eventBus.emit("scene:complete", scenes);

        return story;

    }

    async generateCharacters(story) {

        const names = [
            "Hero",
            "Friend",
            "Villain"
        ];

        story.characters =
            this.character.createMany(names);

        eventBus.emit(
            "characters:complete",
            story.characters
        );

        return story;

    }

    async generateDialogue(story) {

        story.scenes.forEach(scene => {

            scene.dialogue =
                this.dialogue.generate(
                    scene,
                    story.characters
                );

        });

        eventBus.emit(
            "dialogue:complete",
            story.scenes
        );

        return story;

    }

        this.queue = new TaskQueue();

        this.providers = new ProviderManager();

        this.story = new StoryEngine();

        this.screenplay = new ScreenplayEngine();

        this.scene = new SceneEngine();

        this.character = new CharacterEngine();

        this.dialogue = new DialogueEngine();

        this.prompt = new PromptEngine();

        this.initialized = false;

    }

    async initialize() {

        if (this.initialized)
            return;

        console.log("Initializing Cabangile AI Engine...");

        eventBus.emit("ai:init");

        this.initialized = true;

    }

    getStatus() {

        return {

            initialized: this.initialized,

            providerCount: this.providers.getProviderNames().length,

            queueSize: this.queue.size(),

            queueRunning: this.queue.isRunning()

        };

    }

}
