export class AIControlsView {
  constructor(api, state, customArchitectures = null) {
    this.api = api;
    this.state = state;
    this.customArchitectures = customArchitectures; // Allow override of architectures
    // Elements
    this.checkbox = document.getElementById('ai-run-checkbox');
    this.aiStatus = document.getElementById('ai-status');
    this.archSelect = document.getElementById('ai-arch');
    this.budgetInput = document.getElementById('ai-budget');
    this.resizeInput = document.getElementById('ai-resize');
    this.maskWeightInput = document.getElementById('ai-mask-weight');
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
      this.checkbox.addEventListener('change', async () => {
        this.state.aiRunning = this.checkbox.checked;
        this.state.config.aiShouldBeRun = this.state.aiRunning;
        Object.assign(this.state.config, this._collectAIConfig());
        this.state.configUpdated = true;
        this.updateInputsDisabled();
        // Immediately persist config to backend when checkbox is changed
        try {
          await this.api.updateConfig(this.state.config);
          this.state.configUpdated = false;
        } catch (e) {
          console.error('Failed to update config on AI checkbox change:', e);
        }
      });
    }

    // Input change handlers: when running (checkbox checked) we update backend with full config on each change.
    const onInputChange = () => {
      // Always queue config locally; callers push at workflow boundaries
      Object.assign(this.state.config, this._collectAIConfig());
      this.state.configUpdated = true;
    };
    [this.archSelect, this.budgetInput, this.resizeInput, this.maskWeightInput].forEach(inp => {
      if (inp) inp.addEventListener('change', onInputChange);
    });

    this.updateInputsDisabled();
  }

  _collectAIConfig() {
    // Determine appropriate default architecture based on whether custom architectures are used
    const defaultArch = this.customArchitectures ? this.customArchitectures[0] : 'resnet18';

    const resolveWeight = (inputEl, fallback) => {
      if (!inputEl) return fallback;
      const raw = inputEl.value;
      if (raw === '' || raw === null || typeof raw === 'undefined') {
        return fallback;
      }
      const parsed = Number(raw);
      return Number.isFinite(parsed) ? parsed : fallback;
    };

    const currentMaskWeight = typeof this.state.config.mask_loss_weight === 'number'
      ? this.state.config.mask_loss_weight : 1;

    return {
      aiShouldBeRun: !!(this.checkbox && this.checkbox.checked),
      architecture: this.archSelect?.value || this.state.config.architecture || defaultArch,
      budget: Number(this.budgetInput?.value || this.state.config.budget || 1000),
      resize: Number(this.resizeInput?.value || this.state.config.resize || 224),
      mask_loss_weight: resolveWeight(this.maskWeightInput, currentMaskWeight)
    };
  }

  _applyConfigToInputs(cfg) {
    if (this.archSelect && cfg.architecture) this.archSelect.value = cfg.architecture;
    if (this.budgetInput && cfg.budget !== undefined) this.budgetInput.value = cfg.budget;
    if (this.resizeInput && cfg.resize !== undefined) this.resizeInput.value = cfg.resize;
    if (this.maskWeightInput && cfg.mask_loss_weight !== undefined) this.maskWeightInput.value = cfg.mask_loss_weight;
  }

  updateInputsDisabled() {
    const running = !!(this.checkbox && this.checkbox.checked);
    // When running, disable editing (as per requirement: only change when unchecked)
    [this.archSelect, this.budgetInput, this.resizeInput, this.maskWeightInput].forEach(inp => {
      if (inp) inp.disabled = running;
    });
    if (this.aiStatus) {
      this.aiStatus.textContent = running ? 'AI running' : 'AI stopped';
    }
  }

  async loadArchitectures() {
    if (!this.archSelect) return;
    try {
      // 1) Determine list of architectures
      let archs = [];
      if (this.customArchitectures) {
        archs = this.customArchitectures;
      } else if (this.state?.config && Array.isArray(this.state.config.available_architectures)) {
        archs = this.state.config.available_architectures;
        if (!archs.length) {
          archs = await this.api.getArchitectures();
        }
      } else {
        archs = await this.api.getArchitectures();
      }

      // 2) Rebuild select: always add a disabled, selected placeholder first
      this.archSelect.innerHTML = '';
      const placeholderOpt = document.createElement('option');
      placeholderOpt.value = '';
      placeholderOpt.disabled = true;
      placeholderOpt.selected = true;
      placeholderOpt.textContent = 'Select architecture';
      this.archSelect.appendChild(placeholderOpt);

      // 3) Append options
      archs.forEach(a => {
        const opt = document.createElement('option');
        opt.value = a;
        opt.textContent = a;
        this.archSelect.appendChild(opt);
      });

      // 4) Try to select current config architecture if present in list
      const current = this.state?.config?.architecture;
      if (current && archs.includes(current)) {
        this.archSelect.value = current;
      }
    } catch (e) {
      console.error('Failed to load architectures:', e);
    }
  }

  async persistConfig(partial) {
    // Backwards compat shim: keep only local merge; actual PUT happens elsewhere
    const merged = { ...this.state.config, ...partial };
    this.state.config = merged;
    this.state.configUpdated = true;
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
