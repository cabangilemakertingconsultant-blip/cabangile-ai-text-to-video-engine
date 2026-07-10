/**
 * Cabangile AI Studio
 * Character Engine
 * Version 1.0.0
 */

export default class CharacterEngine {

    create(name = "Unknown Character") {

        return {
            id: crypto.randomUUID(),
            name,
            age: null,
            gender: "",
            role: "Supporting",
            appearance: "",
            personality: "",
            background: "",
            goals: "",
            skills: [],
            relationships: []
        };

    }

    createMany(names = []) {
        return names.map(name => this.create(name));
    }

    update(character, updates = {}) {
        return {
            ...character,
            ...updates
        };
    }

}
