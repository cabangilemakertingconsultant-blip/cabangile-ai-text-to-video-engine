/**
 * Cabangile AI Studio
 * Event Bus
 * Version 1.0.0
 */

export class EventBus {
    constructor() {
        this.events = new Map();
    }

    on(event, listener) {
        if (!this.events.has(event)) {
            this.events.set(event, []);
        }
        this.events.get(event).push(listener);
    }

    emit(event, data = null) {
        if (!this.events.has(event)) return;

        this.events.get(event).forEach(listener => {
            try {
                listener(data);
            } catch (error) {
                console.error(`Event Error: ${event}`, error);
            }
        });
    }

    off(event, listener) {
        if (!this.events.has(event)) return;

        const listeners = this.events.get(event);
        this.events.set(
            event,
            listeners.filter(l => l !== listener)
        );
    }

    clear() {
        this.events.clear();
    }
}

const eventBus = new EventBus();

export default eventBus;
