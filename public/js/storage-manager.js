/**
 * Cabangile AI Studio
 * Storage Manager
 * Version 1.0.0
 */

const PROJECT_KEY = "cabangile-ai-project";

export default class StorageManager {

    save(project) {
        try {
            localStorage.setItem(
                PROJECT_KEY,
                JSON.stringify(project)
            );

            return true;

        } catch (error) {
            console.error("Save failed:", error);
            return false;
        }
    }

    load() {

        try {

            const data = localStorage.getItem(PROJECT_KEY);

            if (!data) return null;

            return JSON.parse(data);

        } catch (error) {

            console.error("Load failed:", error);

            return null;
        }

    }

    remove() {

        localStorage.removeItem(PROJECT_KEY);

    }

    exists() {

        return localStorage.getItem(PROJECT_KEY) !== null;

    }

    clearAll() {

        localStorage.clear();

    }

}
