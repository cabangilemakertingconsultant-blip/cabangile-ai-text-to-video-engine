/**
 * Cabangile AI Studio
 * Utility Functions
 * Version 1.0.0
 */

export function generateId(prefix = "id") {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function formatDate(date = new Date()) {
    return new Intl.DateTimeFormat("en-ZA", {
        dateStyle: "medium",
        timeStyle: "short"
    }).format(date);
}

export function isEmpty(value) {
    return (
        value === null ||
        value === undefined ||
        String(value).trim() === ""
    );
}

export function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

export function deepClone(object) {
    return structuredClone(object);
}

export function log(message, data = null) {
    console.log(`[Cabangile AI Studio] ${message}`, data ?? "");
}

export function error(message, err = null) {
    console.error(`[Cabangile AI Studio] ${message}`, err ?? "");
}
