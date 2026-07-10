/**
 * Cabangile AI Studio
 * Dialogue Engine
 * Version 1.0.0
 */

export default class DialogueEngine {

    generate(scene, characters = []) {

        const speaker = characters[0]?.name || "Narrator";

        return [
            {
                id: crypto.randomUUID(),
                speaker,
                emotion: "Neutral",
                text: "This is the beginning of the scene."
            }
        ];

    }

    add(dialogue, line) {
        dialogue.push({
            id: crypto.randomUUID(),
            ...line
        });

        return dialogue;
    }

    remove(dialogue, id) {
        return dialogue.filter(item => item.id !== id);
    }

}
