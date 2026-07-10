/**
 * Cabangile AI Studio
 * Provider Manager
 * Version 1.0.0
 */

export default class ProviderManager {

    constructor() {
        this.providers = new Map();
        this.activeProvider = null;
    }

    register(name, provider) {
        this.providers.set(name, provider);

        if (!this.activeProvider) {
            this.activeProvider = name;
        }
    }

    setActive(name) {
        if (!this.providers.has(name)) {
            throw new Error(`Unknown provider: ${name}`);
        }

        this.activeProvider = name;
    }

    getActive() {
        return this.providers.get(this.activeProvider);
    }

    getProviderNames() {
        return [...this.providers.keys()];
    }

    async generate(type, payload) {

        const provider = this.getActive();

        if (!provider) {
            throw new Error("No AI provider registered.");
        }

        if (typeof provider.generate !== "function") {
            throw new Error("Provider does not implement generate().");
        }

        return await provider.generate(type, payload);
    }

}
