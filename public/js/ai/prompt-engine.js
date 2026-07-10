/**
 * Cabangile AI Studio
 * Prompt Engine
 * Version 1.0.0
 */

export default class PromptEngine {

    createImagePrompt(scene) {

        return `
Cinematic ${scene.title}.
Location: ${scene.location}
Lighting: ${scene.lighting}
Camera: ${scene.camera.shot}
Style: Ultra realistic, 8K, film quality.
`.trim();

    }

    createVideoPrompt(scene) {

        return `
Create a cinematic video.

Scene:
${scene.title}

Camera:
${scene.camera.shot}

Movement:
${scene.camera.movement}

Lighting:
${scene.lighting}

Duration:
${scene.duration} seconds
`.trim();

    }

    createVoicePrompt(character, dialogue) {

        return `
Voice Actor:
${character.name}

Emotion:
${dialogue.emotion}

Dialogue:
${dialogue.text}
`.trim();

    }

}
