/**
 * Cabangile AI Studio
 * AI Task Queue
 * Version 1.0.0
 */

export default class TaskQueue {

    constructor() {
        this.queue = [];
        this.running = false;
    }

    add(task) {
        if (typeof task !== "function") {
            throw new Error("Task must be a function.");
        }

        this.queue.push(task);

        if (!this.running) {
            this.process();
        }
    }

    async process() {

        this.running = true;

        while (this.queue.length > 0) {

            const task = this.queue.shift();

            try {
                await task();
            } catch (error) {
                console.error("Task Queue Error:", error);
            }

        }

        this.running = false;
    }

    clear() {
        this.queue = [];
    }

    size() {
        return this.queue.length;
    }

    isRunning() {
        return this.running;
    }

}
