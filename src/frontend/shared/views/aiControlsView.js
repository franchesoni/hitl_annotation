export class AIControlsView {
  constructor(api, state) {
    this.api = api;
    this.state = state;
    // Elements
    this.checkbox = document.getElementById('ai-run-checkbox');
    this.aiStatus = document.getElementById('ai-status');
    this.archInput = document.getElementById('ai-arch');
    this.archDatalist = document.getElementById('arch-options');
    this.sleepInput = document.getElementById('ai-sleep');
    this.budgetInput = document.getElementById('ai-budget');
    this.resizeInput = document.getElementById('ai-resize');
    this.exportDBBtn = document.getElementById('export-db-btn');

    if (this.exportDBBtn) {
      this.exportDBBtn.addEventListener('click', () => this.api.exportDB());
    }

    // Initialize state mapping
    if (typeof this.state.aiRunning === 'undefined') {
      // derive from config if available
      this.state.aiRunning = !!this.state.config.aiShouldBeRun;
    }

    // Populate inputs with existing config values if present
    this._applyConfigToInputs(this.state.config);

    // Load architectures list
    this.loadArchitectures();

    // Checkbox logic
    if (this.checkbox) {
      this.checkbox.checked = !!this.state.aiRunning;
      this.checkbox.addEventListener('change', () => {
        this.state.aiRunning = this.checkbox.checked;
        this.state.config.aiShouldBeRun = this.state.aiRunning;
        this.state.configUpdated = true;
        // If turning OFF (unchecked), immediately persist only the flag
        // If turning ON, persist full set of options alongside the flag
        if (!this.state.aiRunning) {
          this.persistConfig({ aiShouldBeRun: false });
        } else {
          this.persistConfig(this._collectAIConfig());
        }
        this.updateInputsDisabled();
      });
    }

    // Input change handlers: when running (checkbox checked) we update backend with full config on each change.
    const onInputChange = () => {
      if (this.checkbox && this.checkbox.checked) {
        this.persistConfig(this._collectAIConfig());
      } else {
        // Just keep local until run is enabled
        Object.assign(this.state.config, this._collectAIConfig());
        this.state.configUpdated = true; // for consistency
      }
    };
    [this.archInput, this.sleepInput, this.budgetInput, this.resizeInput].forEach(inp => {
      if (inp) inp.addEventListener('change', onInputChange);
    });

    this.updateInputsDisabled();
  }

  _collectAIConfig() {
    return {
      aiShouldBeRun: !!(this.checkbox && this.checkbox.checked),
      architecture: this.archInput?.value || this.state.config.architecture || 'resnet18',
      sleep: Number(this.sleepInput?.value || this.state.config.sleep || 0),
      budget: Number(this.budgetInput?.value || this.state.config.budget || 1000),
      resize: Number(this.resizeInput?.value || this.state.config.resize || 224)
    };
  }

  _applyConfigToInputs(cfg) {
    if (this.archInput && cfg.architecture) this.archInput.value = cfg.architecture;
    if (this.sleepInput && cfg.sleep !== undefined) this.sleepInput.value = cfg.sleep;
    if (this.budgetInput && cfg.budget !== undefined) this.budgetInput.value = cfg.budget;
    if (this.resizeInput && cfg.resize !== undefined) this.resizeInput.value = cfg.resize;
  }

  updateInputsDisabled() {
    const running = !!(this.checkbox && this.checkbox.checked);
    // When running, disable editing (as per requirement: only change when unchecked)
    [this.archInput, this.sleepInput, this.budgetInput, this.resizeInput].forEach(inp => {
      if (inp) inp.disabled = running;
    });
    if (this.aiStatus) {
      this.aiStatus.textContent = running ? 'AI running' : 'AI stopped';
    }
  }

  async loadArchitectures() {
    if (!this.archDatalist) return;
    try {
      let archs = [];
      if (this.state.config && Array.isArray(this.state.config.available_architectures)) {
        archs = this.state.config.available_architectures;
      }
      if (!archs.length) {
        archs = await this.api.getArchitectures();
      }
      this.archDatalist.innerHTML = '';
      archs.forEach(a => {
        const opt = document.createElement('option');
        opt.value = a;
        this.archDatalist.appendChild(opt);
      });
    } catch (e) {
      console.error('Failed to load architectures:', e);
    }
  }

  async persistConfig(partial) {
    // Merge with existing config then send
    const merged = { ...this.state.config, ...partial };
    this.state.config = merged; // keep local copy consistent
    this.state.configUpdated = true;
    try {
      await this.api.updateConfig(merged);
      this.state.configUpdated = false;
    } catch (e) {
      console.error('Failed to update AI config:', e);
      if (this.aiStatus) this.aiStatus.textContent = 'Update failed';
    }
  }

  render(cfg) {
    // Optional external call to refresh UI with latest config
    if (cfg) {
      this.state.config = { ...this.state.config, ...cfg };
      this.state.configUpdated = false;
      this._applyConfigToInputs(this.state.config);
    }
    if (this.checkbox) {
      this.checkbox.checked = !!this.state.config.aiShouldBeRun;
    }
    this.updateInputsDisabled();
  }
}
