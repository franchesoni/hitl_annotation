// Minimal hotkey manager to avoid duplicate global keydown handlers.
// Usage:
//   const hk = new Hotkeys();
//   hk.bind('ctrl+e', () => exportDB());
//   hk.bind('n', () => next());
//   hk.attach();
// Supports: ctrl/meta/shift/alt modifiers and single keys.

export class Hotkeys {
  constructor(target = document) {
    this.target = target;
    this.handlers = new Map(); // normalized key -> handler
    this.listener = this._onKeydown.bind(this);
    this.attached = false;
  }
  normalize(combo) {
    const parts = String(combo).toLowerCase().split('+').map(s => s.trim()).filter(Boolean);
    const key = parts.pop();
    const mods = new Set(parts.sort());
    return `${mods.has('ctrl') ? 'ctrl+' : ''}${mods.has('meta') ? 'meta+' : ''}${mods.has('alt') ? 'alt+' : ''}${mods.has('shift') ? 'shift+' : ''}${key}`;
  }
  bind(combo, handler) {
    this.handlers.set(this.normalize(combo), handler);
    return this;
  }
  unbind(combo) {
    this.handlers.delete(this.normalize(combo));
    return this;
  }
  attach() {
    if (this.attached) return this;
    this.target.addEventListener('keydown', this.listener);
    this.attached = true;
    return this;
  }
  detach() {
    if (!this.attached) return this;
    this.target.removeEventListener('keydown', this.listener);
    this.attached = false;
    return this;
  }
  _onKeydown(e) {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable)) return;
    const parts = [];
    if (e.ctrlKey) parts.push('ctrl');
    if (e.metaKey) parts.push('meta');
    if (e.altKey) parts.push('alt');
    if (e.shiftKey) parts.push('shift');
    const key = (e.key || '').toLowerCase();
    const norm = this.normalize([...parts, key].join('+'));
    const handler = this.handlers.get(norm);
    if (handler) {
      e.preventDefault();
      try { handler(e); } catch (err) { console.error('Hotkey handler error:', err); }
    }
  }
}

