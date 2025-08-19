// aiControlsView.js - Handles AI run/stop controls and config

export class AIControlsView {
  constructor(api, state) {
    this.api = api;
    this.state = state;
    this.runBtn = document.getElementById('run-ai-btn');
    this.stopBtn = document.getElementById('stop-ai-btn');
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

    if (this.runBtn) {
      this.runBtn.addEventListener('click', async () => {
        try {
          const res = await this.api.runAI({
            architecture: this.archInput?.value || 'resnet18',
            sleep: Number(this.sleepInput?.value || 0),
            budget: Number(this.budgetInput?.value || 1000),
            resize: Number(this.resizeInput?.value || 64)
          });
          this.state.aiRunning = true;
          if (this.aiStatus) this.aiStatus.textContent = res.status;
        } catch (e) {
          if (this.aiStatus) this.aiStatus.textContent = e.message;
          this.state.aiRunning = e.message.includes('already');
          console.error('Failed to start AI:', e);
        } finally {
          this.updateAIButtons();
        }
      });
    }

    if (this.stopBtn) {
      this.stopBtn.addEventListener('click', async () => {
        try {
          const res = await this.api.stopAI();
          this.state.aiRunning = false;
          if (this.aiStatus) this.aiStatus.textContent = res.status;
        } catch (e) {
          if (this.aiStatus) this.aiStatus.textContent = e.message;
          this.state.aiRunning = !e.message.includes('not running');
          console.error('Failed to stop AI:', e);
        } finally {
          this.updateAIButtons();
        }
      });
    }

    this.loadArchitectures();
    this.updateAIButtons();
  }

  updateAIButtons() {
    if (this.runBtn) this.runBtn.style.display = this.state.aiRunning ? 'none' : 'inline-block';
    if (this.stopBtn) this.stopBtn.style.display = this.state.aiRunning ? 'inline-block' : 'none';
  }

  async loadArchitectures() {
    if (this.archDatalist) {
      try {
        const archs = await this.api.getArchitectures();
        archs.forEach(a => {
          const opt = document.createElement('option');
          opt.value = a;
          this.archDatalist.appendChild(opt);
        });
      } catch (e) {
        console.error('Failed to load architectures:', e);
      }
    }
  }
}
