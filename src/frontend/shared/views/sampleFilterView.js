export function applySampleFilterConfig(state, cfg) {
  if (!state || !state.config || !cfg) {
    return { value: '', count: null };
  }
  const value = (typeof cfg.sample_path_filter === 'string') ? cfg.sample_path_filter : '';
  const count = (typeof cfg.sample_path_filter_count === 'number') ? cfg.sample_path_filter_count : null;
  state.config.samplePathFilter = value;
  state.config.sampleFilterCount = count;
  return { value, count };
}

export class SampleFilterView {
  constructor({ inputEl, countEl, api, state, onConfigApplied = null } = {}) {
    this.inputEl = inputEl || null;
    this.countEl = countEl || null;
    this.api = api || null;
    this.state = state || { config: {} };
    this.onConfigApplied = onConfigApplied;
    this.editing = false;
    this.commitInFlight = false;

    if (this.inputEl) {
      this.inputEl.readOnly = true;
      this.inputEl.addEventListener('focus', () => this._handleFocus());
      this.inputEl.addEventListener('blur', () => this._handleBlur());
      this.inputEl.addEventListener('keydown', (evt) => this._handleKeydown(evt));
    }

    this.render();
  }

  applyServerConfig(cfg) {
    applySampleFilterConfig(this.state, cfg);
    this.render();
  }

  render() {
    if (this.inputEl) {
      if (!this.editing && !this.commitInFlight) {
        this.inputEl.value = this.state?.config?.samplePathFilter || '';
      }
      this.inputEl.readOnly = !(this.editing && !this.commitInFlight);
      this.inputEl.disabled = !!this.commitInFlight;
    }
    if (this.countEl) {
      const count = this.state?.config?.sampleFilterCount;
      if (typeof count === 'number') {
        const noun = count === 1 ? 'match' : 'matches';
        this.countEl.textContent = `${count} ${noun}`;
      } else {
        this.countEl.textContent = '';
      }
    }
  }

  _handleFocus() {
    if (this.commitInFlight) return;
    this.editing = true;
    if (this.inputEl) this.inputEl.readOnly = false;
  }

  _handleBlur() {
    if (this.commitInFlight) return;
    this.editing = false;
    if (this.inputEl) {
      this.inputEl.readOnly = true;
      this.inputEl.value = this.state?.config?.samplePathFilter || '';
    }
    this.render();
  }

  _handleKeydown(evt) {
    if (!evt) return;
    if (evt.key === 'Enter') {
      evt.preventDefault();
      this._submit().catch(() => {});
    } else if (evt.key === 'Escape') {
      evt.preventDefault();
      if (this.inputEl) {
        this.inputEl.value = this.state?.config?.samplePathFilter || '';
        this.inputEl.blur();
      }
    }
  }

  async _submit() {
    if (!this.inputEl || !this.api || this.commitInFlight) return;
    const current = this.state?.config?.samplePathFilter || '';
    const trimmed = (this.inputEl.value ?? '').trim();
    if (trimmed === current) {
      this.editing = false;
      this.render();
      this.inputEl.blur();
      return;
    }
    this.commitInFlight = true;
    this.render();
    try {
      const cfg = await this.api.updateConfig({ samplePathFilter: trimmed || null });
      this.applyServerConfig(cfg);
      if (typeof this.onConfigApplied === 'function') {
        this.onConfigApplied(cfg);
      }
    } catch (err) {
      console.error('Failed to update sample path filter:', err);
      alert('Failed to update sample filter. See console for details.');
      if (this.inputEl) {
        this.inputEl.value = current;
      }
    } finally {
      this.commitInFlight = false;
      this.editing = false;
      if (this.inputEl) {
        this.inputEl.blur();
      }
      this.render();
    }
  }
}
