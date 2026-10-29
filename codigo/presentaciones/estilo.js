// Tema, layouts y piezas de la presentación de avance de VotoGrafo (las slides están en avance.js).
// Imitan la propuesta que el grupo armó en Canva (proyecto/1_propuesta/VotoGrafo_propuesta.pdf): portada azul noche con red
// tenue y hemiciclo, contenido sobre gris lavanda con tarjetas blancas, títulos con serif y acento turquesa.
// La esquina superior derecha queda libre en todas las slides para el video del integrante que habla.
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const RAIZ = path.resolve(__dirname, "..", "..");  // carpeta de la entrega
const RES = JSON.parse(fs.readFileSync(path.join(RAIZ, "resultados", "resultados.json"), "utf8"));
const FIG = (f) => path.join(RAIZ, "figuras", f);

// Canva usa Lora y Lato; aquí Cambria y Calibri, que vienen con Office y se ven parecidas
const THEME = {
  name: "VotoGrafo",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "13203A", lt1: "FFFFFF", dk2: "4A5568", lt2: "EEF0F6",
    accent1: "0F7C83", accent2: "2E7D4F", accent3: "E0A100",
    accent4: "C8102E", accent5: "6A4C93", accent6: "1982C4",
    hlink: "1982C4", folHlink: "6A4C93",
  },
};
// hex para las opciones que no aceptan colores de esquema
const HEX = { tinta: "13203A", texto: "4A5568", gris: "8A94A6", grisClaro: "D5D9E2", fondo: "EEF0F6",
  turquesa: "0F7C83", verde: "2E7D4F", ambar: "E0A100", rojo: "C8102E", morado: "6A4C93", azul: "1982C4" };

const INTEGRANTES = "Ignacio Garrido · Andrés Infante · Sofía Schele · Ignacio Sufán · Alinna Vidal";
const PIE = "VotoGrafo · IIC3641 GML 2026-2";

function nuevo(titulo) {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = titulo;
  pres.author = "Grupo VotoGrafo";
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "PORTADA",
    background: { path: FIG("fondo_oscuro.png") },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.55, w: 9.6, h: 2.0,
        fontSize: 46, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", align: "left",
        margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.8, w: 7.4, h: 1.5,
        fontSize: 20, color: C.background2, valign: "top", align: "left", margin: 0 }, text: "" } },
      { text: { text: "Pontificia Universidad Católica de Chile", options: { x: 0.8, y: 6.75, w: 6, h: 0.3,
        fontSize: 12, color: C.background2, margin: 0 } } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENIDO",
    background: { path: FIG("fondo_claro.png") },
    margin: [0.5, 0.7, 0.6, 0.7],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.45, w: 10.3, h: 0.95,
        fontSize: 34, bold: true, color: C.text1, fontFace: THEME.headFontFace, valign: "middle", align: "left",
        margin: 0 }, text: "" } },
      { text: { text: PIE, options: { x: 0.7, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: C.text2, margin: 0 } } },
    ],
    slideNumber: { x: 12.15, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: HEX.texto, align: "right" },
  });
  pres.defineSlideMaster({
    title: "OSCURO",
    background: { path: FIG("fondo_oscuro.png") },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.45, w: 10.3, h: 0.95,
        fontSize: 34, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "middle", align: "left",
        margin: 0 }, text: "" } },
      { text: { text: PIE, options: { x: 0.7, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: "C9D3E6", margin: 0 } } },
    ],
    slideNumber: { x: 12.15, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: "C9D3E6", align: "right" },
  });
  return { pres, C };
}

// Portada: tres puntos de voto, título, subtítulo, integrantes y hemiciclo
function portada(pres, C, sec, titulo, subtitulo) {
  const s = pres.addSlide({ masterName: "PORTADA", sectionTitle: sec });
  puntos(s, C, 0.8, 0.75, 0.24);
  s.addText(titulo, { placeholder: "title" });
  s.addText([{ text: subtitulo, options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: INTEGRANTES, options: { fontSize: 16 } }], { placeholder: "body" });
  s.addImage({ path: FIG("hemiciclo.png"), x: 7.0, y: 4.1, w: 5.9, h: 3.2, transparency: 0, objectName: "hemiciclo" });
  return s;
}

// Puntos de voto (motivo de Canva): verde, rojo, ámbar
function puntos(slide, C, x, y, d = 0.16) {
  [C.accent2, C.accent4, C.accent3].forEach((col, i) => {
    slide.addShape("ellipse", { x: x + i * (d + 0.1), y, w: d, h: d, fill: { color: col }, line: { color: col },
      objectName: `punto-${i}` });
  });
}

// Círculo con número o signo, como los pasos del plan de Canva
function nodo(slide, C, x, y, txt, color, d = 0.55, tam = 16) {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: color || C.accent1 }, line: { color: color || C.accent1 },
    objectName: `nodo-${txt}` });
  slide.addText(txt, { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: tam, bold: true,
    color: C.background1, margin: 0, isTextBox: true });
}

// Tarjeta blanca con sombra suave (las de Canva); oscura=true para el recuadro azul noche
function tarjeta(slide, C, x, y, w, h, nombre, oscura = false) {
  slide.addShape("roundRect", { x, y, w, h, rectRadius: 0.1, fill: { color: oscura ? C.text1 : C.background1 },
    line: { color: oscura ? C.text1 : C.background1 }, objectName: nombre || "tarjeta",
    shadow: { type: "outer", color: HEX.tinta, opacity: 0.08, blur: 8, offset: 2, angle: 90 } });
}

// Cifra grande en serif con rótulo a la derecha (tarjetas de la slide 2 de Canva) o debajo
function cifra(slide, C, x, y, w, numero, rotulo, color, { lado = false, tam = 40 } = {}) {
  if (lado) {
    slide.addText(numero, { x, y, w: w * 0.42, h: 1.0, fontSize: tam, bold: true, color: color || C.accent1,
      fontFace: THEME.headFontFace, valign: "middle", margin: 0, isTextBox: true });
    slide.addText(rotulo, { x: x + w * 0.45, y, w: w * 0.55, h: 1.0, fontSize: 15, color: C.text1, valign: "middle",
      margin: 0, isTextBox: true });
  } else {
    slide.addText(numero, { x, y, w, h: 0.8, fontSize: tam, bold: true, color: color || C.accent1,
      fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
    slide.addText(rotulo, { x, y: y + 0.82, w, h: 0.75, fontSize: 14, color: C.text2, margin: 0, valign: "top",
      isTextBox: true });
  }
}

// Tabla con encabezado azul noche y filas blancas; resaltar = índices de fila (desde 1) en negrita turquesa
function tabla(slide, filas, opts, resaltar = [], colorRes = HEX.turquesa) {
  const rows = filas.map((f, i) => f.map((c, j) => ({ text: c, options: {
    bold: i === 0 || resaltar.includes(i), color: i === 0 ? "FFFFFF" : (resaltar.includes(i) ? colorRes : HEX.tinta),
    fill: { color: i === 0 ? HEX.tinta : (i % 2 ? "FFFFFF" : "F6F7FB") }, align: j === 0 ? "left" : "center",
    fontSize: opts.fontSize || 14, valign: "middle" } })));
  slide.addTable(rows, { border: { type: "solid", pt: 0.5, color: "E3E6EE" }, margin: 0.07, ...opts });
}

// Formato chileno: coma decimal y punto de miles
const fmt = (x, d = 3) => Number(x).toFixed(d).replace(".", ",");
const miles = (x) => String(Math.round(Number(x))).replace(/\B(?=(\d{3})+(?!\d))/g, ".");
const pct = (x, d = 0) => (100 * Number(x)).toFixed(d).replace(".", ",") + "%";
const pts = (x) => (100 * Number(x)).toFixed(1).replace(".", ",");

function ejeChart() {
  return {
    catAxisLabelColor: HEX.tinta, valAxisLabelColor: HEX.texto, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    dataLabelFontFace: "+mn-lt", dataLabelFontSize: 12, dataLabelColor: HEX.tinta,
    valGridLine: { color: "E3E6EE", size: 0.75 }, catGridLine: { style: "none" },
    titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: HEX.tinta,
    legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.tinta,
    plotArea: { fill: { color: "FFFFFF" } },
  };
}

async function guardar(pres, destino) {
  const { applyTheme } = require(process.env.APPLY_THEME);
  fs.mkdirSync(path.dirname(destino), { recursive: true });
  await pres.writeFile({ fileName: destino });
  await applyTheme(destino, THEME);
  console.log("escrito", destino);
}

module.exports = { nuevo, portada, puntos, nodo, tarjeta, cifra, tabla, fmt, miles, pct, pts, ejeChart, guardar,
  RES, FIG, RAIZ, HEX, INTEGRANTES, THEME };
