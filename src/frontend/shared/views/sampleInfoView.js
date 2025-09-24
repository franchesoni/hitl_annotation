export class SampleInfoView {
  constructor({ container, state } = {}) {
    this.container = typeof container === 'string' ? document.querySelector(container) : container;
    if (!this.container) {
      throw new Error('SampleInfoView: container not found');
    }
    this.state = state || {};
    this.sampleId = null;
    this.filepath = null;
    this.render();
  }

  update({ sampleId, filepath }) {
    if (typeof sampleId !== 'undefined') {
      this.sampleId = sampleId;
    }
    if (typeof filepath !== 'undefined') {
      this.filepath = filepath;
    }
    this.render();
  }

  render() {
    const idDisplay = (this.sampleId !== null && this.sampleId !== undefined) ? this.sampleId : '—';
    const pathDisplay = this.filepath || '—';
    this.container.innerHTML = '';
    const idDiv = document.createElement('div');
    idDiv.className = 'sample-info-line';
    idDiv.textContent = `ID: ${idDisplay}`;
    const pathDiv = document.createElement('div');
    pathDiv.className = 'sample-info-line';
    pathDiv.textContent = `File: ${pathDisplay}`;
    pathDiv.title = this.filepath || '';
    this.container.appendChild(idDiv);
    this.container.appendChild(pathDiv);
  }
}
