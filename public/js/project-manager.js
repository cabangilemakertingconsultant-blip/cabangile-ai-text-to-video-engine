/**
 * Cabangile AI Studio
 * Project Manager
 * Version 1.0.0
 */

import StorageManager from "./storage-manager.js";
import { generateId, formatDate } from "./utils.js";

export default class ProjectManager {

    constructor() {
        this.storage = new StorageManager();
        this.project = null;
    }

    create(name = "Untitled Project") {

        this.project = {
            id: generateId("project"),
            name,
            created: formatDate(),
            updated: formatDate(),
            story: "",
            scenes: [],
            characters: [],
            timeline: []
        };

        return this.project;
    }

    save() {

        this.project.updated = formatDate();

        return this.storage.save(this.project);

    }

    load() {

        this.project = this.storage.load();

        return this.project;

    }

    getProject() {

        return this.project;

    }

    delete() {

        this.storage.remove();

        this.project = null;

    }

}
