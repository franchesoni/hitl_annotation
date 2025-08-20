export class StrategyView {
  constructor() {
    this.strategySelect = document.getElementById('strategy-select');
    this.specificClassSelect = document.getElementById('specific-class-select');
    this.specificClassLabel = document.getElementById('specific-class-label');
    this.currentStrategy = this.strategySelect ? this.strategySelect.value : null;
    this.currentSpecificClass = this.specificClassSelect ? this.specificClassSelect.value : null;
    this.toggleSpecificClassSelect();
    if (this.strategySelect) {
      this.strategySelect.addEventListener('change', () => {
        this.currentStrategy = this.strategySelect.value;
        this.toggleSpecificClassSelect();
      });
    }
    if (this.specificClassSelect) {
      this.specificClassSelect.addEventListener('change', () => {
        this.currentSpecificClass = this.specificClassSelect.value;
      });
    }
  }
  toggleSpecificClassSelect() {
    const show = this.currentStrategy === 'specific_class';
    if (this.specificClassSelect) this.specificClassSelect.style.display = show ? 'inline-block' : 'none';
    if (this.specificClassLabel) this.specificClassLabel.style.display = show ? 'inline-block' : 'none';
  }
  updateClasses(classes) {
    if (this.specificClassSelect) {
      const previous = this.specificClassSelect.value;
      this.specificClassSelect.innerHTML = classes.map(c => `<option value="${c}">${c}</option>`).join('');
      if (classes.includes(previous)) {
        this.specificClassSelect.value = previous;
      } else if (classes.length > 0) {
        this.specificClassSelect.selectedIndex = 0;
      }
      this.currentSpecificClass = this.specificClassSelect.value || null;
    }
  }
}
