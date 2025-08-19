// trainingCurveView.js - Displays training curve

export class TrainingCurveView {
  constructor(api) {
    this.api = api;
    this.canvas = document.getElementById('training-curve');
    this.trainingRequestId = 0;
  }

  async update() {
    if (!this.canvas) return;
    const requestId = ++this.trainingRequestId;
    try {
      const data = await this.api.getTrainingStats();
      if (requestId !== this.trainingRequestId) return;
      const points = data.map(d => ({ x: d.epoch, y: d.accuracy ?? 0 }));
      drawCurve(this.canvas, points);
    } catch (e) {
      if (requestId === this.trainingRequestId) {
        console.error('Failed to fetch training stats:', e);
      }
    }
  }
}

function drawCurve(canvas, points) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!points || points.length === 0) return;
  const padding = 30;
  const maxX = points[points.length - 1].x || 1;
  const maxY = 1;

  ctx.strokeStyle = '#ccc';
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, h - padding);
  ctx.lineTo(w - padding, h - padding);
  ctx.stroke();

  ctx.fillStyle = '#e0e0e0';
  ctx.font = '12px Roboto, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Epoch', w / 2, h - 5);
  ctx.save();
  ctx.translate(padding - 25, h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Val Accuracy', 0, 0);
  ctx.restore();

  const yTicks = [0, 0.5, 1];
  yTicks.forEach(t => {
    const y = h - padding - t * (h - 2 * padding);
    ctx.beginPath();
    ctx.moveTo(padding - 5, y);
    ctx.lineTo(padding, y);
    ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(t.toString(), padding - 7, y + 4);
  });

  const step = Math.max(1, Math.floor(maxX / 5));
  for (let t = 0; t <= maxX; t += step) {
    const x = padding + (t / maxX) * (w - 2 * padding);
    ctx.beginPath();
    ctx.moveTo(x, h - padding);
    ctx.lineTo(x, h - padding + 5);
    ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillText(t.toString(), x, h - padding + 15);
  }

  ctx.strokeStyle = '#007acc';
  ctx.beginPath();
  points.forEach((p, idx) => {
    const x = padding + (p.x / maxX) * (w - 2 * padding);
    const y = h - padding - (p.y / maxY) * (h - 2 * padding);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}
