/**
 * Cabangile AI Studio
 * UI Controller
 * Version 1.0.0
 */

export default class UIController {

    constructor() {
        this.output = document.getElementById("output");
        this.storyInput = document.getElementById("storyInput");
    }

    showMessage(message) {
        if (this.output) {
            this.output.innerHTML = `<p>${message}</p>`;
        }
    }

    getStory() {
        return this.storyInput ? this.storyInput.value.trim() : "";
    }

    setStory(text) {
        if (this.storyInput) {
            this.storyInput.value = text;
        }
    }

    clearStory() {
        if (this.storyInput) {
            this.storyInput.value = "";
        }
    }

    bindButton(id, handler) {
        const button = document.getElementById(id);

        if (!button) {
            console.warn(`Button '${id}' not found.`);
            return;
        }

        button.addEventListener("click", handler);
    }
}
