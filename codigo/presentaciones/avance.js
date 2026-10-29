// Avance (9/11): video de 5 minutos. Cubre lo que el plan de la propuesta comprometió para esta fecha (pasos 1 a 3:
// datos, grafo, análisis descriptivo, baselines y setting experimental) y el feedback del profesor (NOMINATE como
// baseline; precisión y recall sobre los votos fuera de línea).
const path = require("path");
const { nuevo, portada, nodo, tarjeta, cifra, tabla, fmt, miles, pct, pts, ejeChart, guardar, RES, FIG, RAIZ, HEX } = require("./estilo");
const { escribirGuion } = require("./guion");

const D = RES.datos, A = RES.A, B = RES.B, RUP = RES.rupturas, RM = RUP.modelos, NOM = RES.nominate;
const R = D.red_covotacion, SP = D.split;
const { pres, C } = nuevo("VotoGrafo: avance");
const notas = [];
const nota = (s, t) => { s.addNotes(t); notas.push(t); };
const sec = "Avance";
const f2 = (x) => fmt(x, 2);
const K_PAR = "Mayoría del partido (visibles)", K_COV = "Mayoría de co-votación (visibles)", K_NOM = "NOMINATE (2 dimensiones)";
const K_BASE = "sin caminos tipados";   // la R-GCN base: sin la extensión del decoder
const RB = RES.ablaciones[K_BASE];

// 1. Portada
pres.addSection({ title: sec });
let s = portada(pres, C, sec, "VotoGrafo: ¿cómo votará cada diputado?", "Avance del proyecto · IIC3641 GML 2026-2");
nota(s, `Hola, somos VotoGrafo. Les mostramos el avance de nuestro proyecto: predecir cómo vota cada diputado de la Cámara con redes neuronales de grafos, y sobre todo quién se sale de la línea de su partido.`);

// 2. Plan: qué está listo
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
s.addText("Lo comprometido para hoy está listo", { placeholder: "title" });
const plan = [["Datos y propuesta", "listo", C.accent2], ["Grafo y análisis descriptivo", "listo", C.accent2],
  ["Baselines y setting experimental", "listo", C.accent2], ["R-GCN", "primera versión", C.accent3],
  ["Semillas, ablaciones, subgrupos", "pendiente", HEX.gris]];
s.addShape("line", { x: 1.15, y: 2.33, w: 10.9, h: 0, line: { color: HEX.tinta, width: 1 }, objectName: "linea-plan" });
plan.forEach(([t, est, col], i) => {
  const x = 0.7 + i * 2.48;
  nodo(s, C, x + 0.95, 2.05, String(i + 1), i < 3 ? C.accent1 : (i === 3 ? C.accent3 : HEX.gris));
  tarjeta(s, C, x, 2.9, 2.3, 1.55, `paso-${i}`);
  s.addText(t, { x: x + 0.2, y: 3.0, w: 1.95, h: 0.95, fontSize: 16, bold: true, color: C.text1, fontFace: "Cambria",
    valign: "middle", margin: 0, isTextBox: true });
  s.addText(est, { x: x + 0.2, y: 3.95, w: 1.95, h: 0.35, fontSize: 13, bold: true, color: col, margin: 0, isTextBox: true });
});
tarjeta(s, C, 0.7, 4.9, 11.9, 1.75, "feedback", true);
s.addText("Feedback de la propuesta, incorporado", { x: 1.0, y: 5.05, w: 11.3, h: 0.45, fontSize: 17, bold: true,
  color: "FFFFFF", fontFace: "Cambria", margin: 0, isTextBox: true });
s.addText([{ text: "Nuevo baseline de puntos ideales: NOMINATE, el estándar para completar votaciones nominales.", options: { bullet: true, breakLine: true } },
  { text: "Además de accuracy, precisión y recall sobre los votos fuera de línea en todos los modelos.", options: { bullet: true } }],
  { x: 1.0, y: 5.5, w: 11.3, h: 1.0, fontSize: 15, color: "E6EAF2", paraSpaceAfter: 4, margin: 0, isTextBox: true });
nota(s, `En la propuesta nos comprometimos a llegar a hoy con los datos, el grafo, el análisis descriptivo, los baselines y el setting experimental. Está todo listo, y además entrenamos una primera R-GCN. También incorporamos el feedback del profesor: agregamos NOMINATE, un modelo de puntos ideales que es el estándar en ciencia política para completar votaciones, y ahora reportamos precisión y recall sobre los votos fuera de línea, no solo accuracy.`);

// 3. Datos
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
s.addText("Datos abiertos de la Cámara, 2022-2026", { placeholder: "title" });
const pasos = [["API", "opendata.camara.cl, servicios legislativo y diputados"], ["Caché", `${miles(D.votaciones_total_periodo)} votaciones guardadas en XML`],
  ["CSV", "votos, votaciones, proyectos, autores, militancias"], ["Limpieza", "proyectos de ley; partido vigente ese día"]];
pasos.forEach(([t, d], i) => {
  const x = 0.7 + i * 3.0;
  tarjeta(s, C, x, 1.85, 2.7, 1.7, `paso-${i}`);
  nodo(s, C, x + 0.2, 2.0, String(i + 1), C.accent1, 0.45, 13);
  s.addText(t, { x: x + 0.8, y: 2.0, w: 1.8, h: 0.45, fontSize: 17, bold: true, color: C.text1, fontFace: "Cambria", valign: "middle", margin: 0, isTextBox: true });
  s.addText(d, { x: x + 0.2, y: 2.6, w: 2.35, h: 0.85, fontSize: 14, color: C.text2, valign: "top", margin: 0, isTextBox: true });
});
const cifras = [[miles(D.votaciones_proyecto_ley), "votaciones de proyectos de ley"], [miles(D.votos), "votos individuales en 3 clases"],
  [String(D.proyectos), "proyectos con texto, autores y comisión"], [String(D.diputados), `diputados de ${D.partidos} partidos`]];
cifras.forEach(([n, t], i) => {
  const x = 0.7 + i * 3.0;
  tarjeta(s, C, x, 3.95, 2.7, 1.75, `cifra-${i}`);
  cifra(s, C, x + 0.25, 4.1, 2.3, n, t, C.accent1, { tam: 34 });
});
s.addText("Se descartan los votos «dispensado». Las votaciones de resolución y «otros» no traen texto ni autores.",
  { x: 0.7, y: 6.05, w: 11.9, h: 0.5, fontSize: 14, italic: true, color: C.text2, margin: 0, isTextBox: true });
nota(s, `Bajamos todo el período 2022 a 2026 desde la API de la Cámara, con caché local para que sea reproducible. Nos quedamos con las votaciones de proyectos de ley, que traen texto, autores y comisión: ${miles(D.votaciones_proyecto_ley)} votaciones y ${miles(D.votos)} votos. El partido de cada voto es la militancia vigente ese día, porque hay diputados que cambian de partido durante el período.`);

// 4. Descriptivo
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
s.addText("Partidos disciplinados, pero no del todo", { placeholder: "title" });
const disc = Object.entries(D.disciplina).sort((a, b) => a[1] - b[1]);
tarjeta(s, C, 0.7, 1.8, 7.3, 4.95, "graf-disciplina");
s.addChart(pres.charts.BAR, [{ name: "disciplina", labels: disc.map((d) => d[0]), values: disc.map((d) => d[1]) }], {
  x: 0.85, y: 1.9, w: 7.0, h: 4.75, barDir: "bar", chartColors: [HEX.turquesa], showValue: true, dataLabelPosition: "outEnd",
  dataLabelFormatCode: "0.00", valAxisMinVal: 0.5, valAxisMaxVal: 1, showLegend: false, showTitle: true,
  title: "Votos iguales a la mayoría del propio partido", ...ejeChart(), catAxisLabelFontSize: 11,
});
const cs = [[pct(D.pct_rupturas), "de los votos de militantes van contra la mayoría de su partido", C.accent5],
  [pct(D.pct_independientes), "de los votos son de independientes: no tienen línea que seguir", C.accent1],
  [pct(D.clases[2]), "de los votos son abstenciones: clase minoritaria", C.accent3]];
cs.forEach(([n, t, col], i) => {
  tarjeta(s, C, 8.35, 1.8 + i * 1.7, 4.25, 1.45, `dato-${i}`);
  cifra(s, C, 8.55, 1.98 + i * 1.7, 3.95, n, t, col, { lado: true, tam: 34 });
});
nota(s, `Los partidos son disciplinados: en casi todos, más del noventa por ciento de los votos coincide con la mayoría del partido. Por eso copiar a la bancada acierta mucho. Pero ${pct(D.pct_rupturas)} de los votos de militantes van contra su partido, y un cuarto de los votos son de independientes. Esos son los votos fuera de línea que queremos predecir. Además las clases están desbalanceadas: solo ${pct(D.clases[2])} son abstenciones.`);

// 5. Red de co-votación
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
s.addText("La co-votación se ordena en dos bloques", { placeholder: "title" });
tarjeta(s, C, 0.7, 1.8, 6.5, 4.95, "red");
s.addImage({ path: FIG("fig_red_covotacion.png"), x: 0.85, y: 1.88, w: 6.2, h: 4.79, objectName: "fig-red" });
const rc = [[f2(R.acuerdo_intra), "acuerdo medio entre diputados del mismo partido", C.accent1],
  [f2(R.acuerdo_inter), "acuerdo medio entre partidos distintos", C.text2],
  [f2(R.homofilia), "de las aristas unen diputados del mismo partido (homofilia)", C.accent5]];
rc.forEach(([n, t, col], i) => {
  tarjeta(s, C, 7.55, 1.8 + i * 1.7, 5.05, 1.45, `rc-${i}`);
  cifra(s, C, 7.75, 1.98 + i * 1.7, 4.7, n, t, col, { lado: true, tam: 34 });
});
nota(s, `Unimos a cada diputado con los ocho con que más coincide en votaciones disputadas. La red se separa en dos bloques, izquierda y derecha. Hay homofilia: diputados del mismo partido coinciden ${f2(R.acuerdo_intra)}, y de partidos distintos ${f2(R.acuerdo_inter)}. Los independientes, en gris, quedan en ambos bloques: el partido no basta para predecirlos. Con esta red armamos un baseline de grafo: votar como los ocho que más coinciden contigo.`);

// 6. Grafo
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
s.addText("El grafo heterogéneo, construido", { placeholder: "title" });
tarjeta(s, C, 0.7, 1.8, 6.1, 4.95, "esquema");
s.addImage({ path: FIG("esquema_grafo.png"), x: 0.9, y: 1.95, w: 5.7, h: 4.23, objectName: "fig-esquema" });
s.addText("5 tipos de nodo; el voto es una arista con 3 tipos", { x: 0.9, y: 6.2, w: 5.7, h: 0.4, fontSize: 13, italic: true,
  color: C.text2, align: "center", margin: 0, isTextBox: true });
tabla(s, [["relación", "aristas"], ["voto: sí, no, abstención", miles(D.votos)], ["votación similar (texto)", miles(D.aristas_similar)],
  ["votación anterior del mismo proyecto", miles(D.aristas_mismo_proyecto)], ["votación de proyecto", miles(D.votaciones_proyecto_ley)],
  ["autoría", miles(D.autorias)], ["proyecto en comisión", miles(D.proyectos)], ["militancia", String(D.diputados)]],
  { x: 7.15, y: 1.85, w: 5.45, colW: [3.75, 1.7], rowH: 0.45 });
s.addText("Las aristas entre votaciones van siempre del pasado al presente.",
  { x: 7.15, y: 5.65, w: 5.45, h: 0.8, fontSize: 14, italic: true, color: C.text2, margin: 0, isTextBox: true });
nota(s, `Este es el grafo, tal como lo propusimos: diputados, votaciones, proyectos, partidos y comisiones, con el voto como arista de tres tipos. Las dos relaciones entre votaciones, por texto parecido y por mismo proyecto, van siempre del pasado al presente, así una votación nueva recibe información de votaciones anteriores sin ver el futuro. Las votaciones llevan como features el texto, el trámite, la comisión y quién patrocina el proyecto.`);

// 7. Setting experimental
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
s.addText("Setting experimental", { placeholder: "title" });
const tramos = [["train", SP.train, 6.3, C.text1], ["val", SP.val, 2.9, HEX.gris], ["test", SP.test, 2.7, C.accent1]];
let x0 = 0.7;
tramos.forEach(([n, t, w, col]) => {
  s.addShape("roundRect", { x: x0, y: 1.85, w: w - 0.06, h: 0.7, rectRadius: 0.08, fill: { color: col }, line: { color: col }, objectName: `tramo-${n}` });
  s.addText(`${n}: ${miles(t.votaciones)} votaciones`, { x: x0 + 0.15, y: 1.85, w: w - 0.3, h: 0.7, fontSize: 15, bold: true, color: "FFFFFF", valign: "middle", margin: 0, isTextBox: true });
  s.addText(`${t.desde} a ${t.hasta}`, { x: x0, y: 2.6, w, h: 0.35, fontSize: 12, color: C.text2, margin: 0, isTextBox: true });
  x0 += w;
});
const esc = [["Completar una votación (principal)", `Cada día de test (${D.dias_test} días) se ve el 50% de los votos de cada votación y todo lo votado antes. Se predice el resto.`],
  ["Antes de votar (secundaria)", "Igual, pero sin ningún voto de la votación: solo texto, autores, comisión e historial."],
  ["Métricas", "Accuracy y F1 macro; accuracy en votos sí/no; precisión, recall y AP de los votos fuera de línea."],
  ["Voto fuera de línea", "Voto oculto de un militante distinto de la mayoría visible de su partido en esa votación."]];
esc.forEach(([t, d], i) => {
  const x = 0.7 + (i % 2) * 6.05, y = 3.25 + Math.floor(i / 2) * 1.75;
  tarjeta(s, C, x, y, 5.85, 1.55, `esc-${i}`);
  s.addText(t, { x: x + 0.25, y: y + 0.15, w: 5.4, h: 0.4, fontSize: 16, bold: true, color: i === 3 ? C.accent5 : C.text1, fontFace: "Cambria", margin: 0, isTextBox: true });
  s.addText(d, { x: x + 0.25, y: y + 0.6, w: 5.4, h: 0.85, fontSize: 14, color: C.text2, valign: "top", margin: 0, isTextBox: true });
});
nota(s, `Entrenamos con el pasado y evaluamos con el último tramo, de julio de 2025 a marzo de 2026. En la tarea principal vemos la mitad de los votos de cada votación y predecimos el resto. Un voto fuera de línea es el de un militante que no vota como la mayoría visible de su partido; en test hay ${miles(RUP.n)}. El recall dice cuántos encuentra cada modelo, y la precisión, cuántas de sus alertas son correctas.`);

// 8. NOMINATE
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
s.addText("Nuevo baseline: NOMINATE", { placeholder: "title" });
tarjeta(s, C, 0.7, 1.8, 5.6, 4.95, "puntos-ideales");
s.addImage({ path: FIG("fig_puntos_ideales.png"), x: 0.85, y: 1.9, w: 5.3, h: 4.75, sizing: { type: "contain", w: 5.3, h: 4.75 }, objectName: "fig-puntos" });
const pasosN = [["Puntos ideales", "Cada diputado es un punto en un mapa de 2 dimensiones, estimado con todos sus votos."],
  ["Línea de corte", "Cada votación tiene un resultado «sí» y uno «no» en el mapa; vota sí quien queda más cerca del sí."],
  ["Completar", "Con el 50% visible se ubica la línea de corte de la votación y se predice el resto."]];
pasosN.forEach(([t, d], i) => {
  const y = 1.8 + i * 1.3;
  nodo(s, C, 6.65, y + 0.1, String(i + 1), C.accent1, 0.5, 14);
  s.addText(t, { x: 7.35, y, w: 5.25, h: 0.4, fontSize: 16, bold: true, color: C.text1, fontFace: "Cambria", margin: 0, isTextBox: true });
  s.addText(d, { x: 7.35, y: y + 0.4, w: 5.25, h: 0.8, fontSize: 14, color: C.text2, valign: "top", margin: 0, isTextBox: true });
});
tarjeta(s, C, 6.65, 5.75, 5.95, 1.0, "nom-nota", true);
s.addText("No modela la abstención (la trata como dato faltante) y no sirve antes de votar: sin votos no hay línea de corte.",
  { x: 6.85, y: 5.8, w: 5.6, h: 0.9, fontSize: 13, color: "E6EAF2", valign: "middle", margin: 0, isTextBox: true });
nota(s, `El baseline nuevo es NOMINATE, de Poole y Rosenthal. Ubica a cada diputado en un mapa de dos dimensiones según cómo vota; a la izquierda se ve cómo se ordenan los partidos. Cada votación traza una línea de corte en ese mapa. Con los votos visibles, NOMINATE ubica esa línea y predice el resto. Lo implementamos en PyTorch, con el mismo protocolo que los demás. No modela la abstención y no sirve antes de votar.`);

// 9. Baselines medidos
s = pres.addSlide({ masterName: "CONTENIDO", sectionTitle: sec });
s.addText("Baselines: NOMINATE es la vara a superar", { placeholder: "title" });
const fil = [[K_PAR, "Mayoría del partido"], [K_COV, "Co-votación (8 vecinos)"], [K_NOM, "NOMINATE, 2 dimensiones"]];
const filasT = [["completar votación, 50% visible", "accuracy", "acc. sí/no", "precisión fuera de línea", "recall fuera de línea"]];
fil.forEach(([k, n]) => filasT.push([n, fmt(B[k].acc), fmt(B[k].acc_sino), k === K_PAR ? "no marca" : fmt(RM[k].precision, 2), fmt(RM[k].recall, 2)]));
filasT.push(["R-GCN base (primera versión)", fmt(RB.B_acc), fmt(RB.B_acc_sino), fmt(RM[K_BASE].precision, 2), fmt(RM[K_BASE].recall, 2)]);
tabla(s, filasT, { x: 0.7, y: 1.85, w: 11.9, colW: [4.3, 1.6, 1.6, 2.2, 2.2], rowH: 0.55, fontSize: 15 }, [3]);
const mA = Object.entries(A).filter(([k]) => k !== "MLP sin grafo" && k !== "R-GCN completo").sort((a, b) => b[1].acc - a[1].acc)[0];
tarjeta(s, C, 0.7, 5.0, 5.85, 1.75, "a-card");
s.addText("Antes de votar", { x: 0.95, y: 5.12, w: 5.4, h: 0.4, fontSize: 16, bold: true, color: C.text1, fontFace: "Cambria", margin: 0, isTextBox: true });
s.addText(`Mejor baseline: ${mA[0].toLowerCase()}, ${fmt(mA[1].acc)}. Siempre «a favor»: ${fmt(A.Mayoritaria.acc)}. R-GCN base: ${fmt(RB.A_acc)}.`,
  { x: 0.95, y: 5.55, w: 5.4, h: 1.1, fontSize: 14, color: C.text2, valign: "top", margin: 0, isTextBox: true });
tarjeta(s, C, 6.75, 5.0, 5.85, 1.75, "lectura", true);
s.addText(`El partido no encuentra ningún voto fuera de línea. NOMINATE encuentra ${pct(RM[K_NOM].recall)} y acierta en ${pct(RM[K_NOM].precision)} de sus alertas.`,
  { x: 7.0, y: 5.1, w: 5.4, h: 1.55, fontSize: 15, color: "FFFFFF", valign: "middle", margin: 0, isTextBox: true });
nota(s, `Estos son los baselines en la tarea principal. Copiar a la mayoría visible del partido acierta ${fmt(B[K_PAR].acc)}, pero no encuentra ningún voto fuera de línea. La co-votación sube a ${fmt(B[K_COV].acc)} y encuentra ${pct(RM[K_COV].recall)}. NOMINATE llega a ${fmt(B[K_NOM].acc)}; pierde algo porque nunca predice abstenciones, y si miramos solo votos a favor y en contra es el mejor, con ${fmt(B[K_NOM].acc_sino)}. Y encuentra ${pct(RM[K_NOM].recall)} de los votos fuera de línea. Nuestra primera R-GCN llega a ${fmt(RB.B_acc)}: le gana al partido, pero no a NOMINATE.`);

// 10. Próximos pasos
s = pres.addSlide({ masterName: "OSCURO", sectionTitle: sec });
s.addText("Hasta la entrega final (23/11)", { placeholder: "title" });
const prox = [["4", "R-GCN completa", "Decoder con caminos tipados (qué vota su bancada y sus vecinos); si no basta, combinarla con NOMINATE."],
  ["5", "Semillas y ablaciones", "5 semillas; quitar una pieza a la vez para ver qué explica la mejora."],
  ["5", "Votos fuera de línea", "Precisión, recall y AP de cada modelo contra NOMINATE y la co-votación."],
  ["5", "Subgrupos", "Independientes, votaciones disputadas, abstenciones; y la tarea antes de votar."]];
prox.forEach(([n, t, d], i) => {
  const y = 1.85 + i * 1.2;
  nodo(s, C, 0.8, y, n, i === 0 ? C.accent3 : C.accent1);
  s.addText(t, { x: 1.6, y: y - 0.05, w: 3.4, h: 0.65, fontSize: 19, bold: true, color: "FFFFFF", fontFace: "Cambria", valign: "middle", margin: 0, isTextBox: true });
  s.addText(d, { x: 5.0, y: y - 0.05, w: 7.6, h: 0.65, fontSize: 15, color: "D5DCE8", valign: "middle", margin: 0, isTextBox: true });
});
nota(s, `Para la entrega final quedan los pasos cuatro y cinco del plan: la R-GCN completa, con un decoder que cuenta lo que vota la bancada y los vecinos de cada diputado, y si eso no basta para superar a NOMINATE, una combinación de los dos; cinco semillas y ablaciones; precisión y recall de los votos fuera de línea contra NOMINATE; y el análisis por subgrupos. Gracias.`);

(async () => {
  await guardar(pres, path.join(RAIZ, "avance.pptx"));
  escribirGuion(path.join(RAIZ, "guion_avance.md"), "Avance (video de 5 minutos)", notas, 300);
})();
