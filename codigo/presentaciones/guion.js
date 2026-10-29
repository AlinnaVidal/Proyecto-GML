// Escribe el guion de una presentación en Markdown con el tiempo estimado por slide.
const fs = require("fs");

const PALABRAS_POR_MINUTO = 140;

function escribirGuion(destino, titulo, notas, segundosMax) {
  let total = 0;
  const filas = notas.map((t, i) => {
    // una cifra con decimales ("0,918") se lee como unas tres palabras
    const tokens = t.split(/\s+/).filter(Boolean);
    const palabras = tokens.reduce((n, w) => n + (/\d[,.]\d/.test(w) ? 3 : 1), 0);
    const seg = Math.round((palabras / PALABRAS_POR_MINUTO) * 60);
    total += seg;
    return `## Slide ${i + 1} (~${seg} s, ${palabras} palabras)\n\n${t}\n`;
  });
  const mmss = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  const cab = `# Guion: ${titulo}\n\nTiempo estimado a ${PALABRAS_POR_MINUTO} palabras por minuto: ${mmss(total)} ` +
    `(límite ${mmss(segundosMax)}). El mismo texto está en las notas del orador de cada slide.\n\n`;
  fs.writeFileSync(destino, cab + filas.join("\n"), "utf8");
  console.log(`guion ${destino}: ${mmss(total)} de ${mmss(segundosMax)}`);
}

module.exports = { escribirGuion };
