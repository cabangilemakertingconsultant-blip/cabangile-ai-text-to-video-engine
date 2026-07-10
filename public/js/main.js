/**
 * ==========================================================
 * Cabangile AI Studio
 * Main Application
 * Version 2.0.0
 * ==========================================================
 */

import AIEngine from "./ai-engine.js";
import UIController from "./ui-controller.js";
import ProjectManager from "./project-manager.js";

const ai = new AIEngine();
const ui = new UIController();
const projects = new ProjectManager();

document.addEventListener("DOMContentLoaded", async () => {

    await ai.initialize();

    ui.showMessage("✅ Cabangile AI Studio Ready");

    ui.bindButton("newProjectBtn", () => {

        const project = projects.create("Untitled Project");

        ui.clearStory();

        ui.showMessage(`New Project Created<br><b>${project.name}</b>`);

    });

    ui.bindButton("saveProjectBtn", () => {

        const story = ui.getStory();

        if (projects.project) {
            projects.project.story = story;
        }

        if (projects.save()) {

            ui.showMessage("💾 Project Saved Successfully");

        } else {

            ui.showMessage("❌ Failed to Save Project");

        }

    });

    ui.bindButton("clearBtn", () => {

        ui.clearStory();

        ui.showMessage("Editor Cleared");

    });

    ui.bindButton("generateBtn", async () => {

        try {

            const prompt = ui.getStory();

            if (!prompt) {

                ui.showMessage("Please enter a movie idea.");

                return;

            }

            ui.showMessage("🧠 AI is generating your project...");

            const project = await ai.generateMovieProject(prompt);

            projects.project = project;

            ui.showMessage(`
                <h2>${project.title}</h2>

                <p><b>Prompt:</b> ${project.prompt}</p>

                <p><b>Scenes:</b> ${project.scenes.length}</p>

                <p><b>Characters:</b> ${project.characters.length}</p>

                <p><b>Image Prompts:</b> ${project.imagePrompts.length}</p>

                <p><b>Video Prompts:</b> ${project.videoPrompts.length}</p>

                <hr>

                <p>✅ Movie Project Generated Successfully.</p>
            `);

        } catch (error) {

            console.error(error);

            ui.showMessage(
                `<span style="color:red">${error.message}</span>`
            );

        }

    });

    ui.bindButton("storyBtn", () =>
        ui.showMessage("Story Engine Ready")
    );

    ui.bindButton("screenplayBtn", () =>
        ui.showMessage("Screenplay Engine Ready")
    );

    ui.bindButton("charactersBtn", () =>
        ui.showMessage("Character Engine Ready")
    );

    ui.bindButton("voiceBtn", () =>
        ui.showMessage("Voice Engine Ready")
    );

    ui.bindButton("imagesBtn", () =>
        ui.showMessage("Image Prompt Engine Ready")
    );

    ui.bindButton("videoBtn", () =>
        ui.showMessage("Video Prompt Engine Ready")
    );

    ui.bindButton("timelineBtn", () =>
        ui.showMessage("Timeline Ready")
    );

    ui.bindButton("renderBtn", () =>
        ui.showMessage("🎬 Rendering will be available in Version 2.")
    );

});
