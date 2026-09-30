const $ = (id) => document.getElementById(id);
const rupiah = (value) => new Intl.NumberFormat('id-ID', {style:'currency', currency:'IDR', maximumFractionDigits:0}).format(value);
const schemas = {
  synthetic: [['luas_m2','Luas bangunan (m²)',20,500,'any'],['jumlah_kamar','Jumlah kamar',1,10,1],['usia_tahun','Usia bangunan (tahun)',0,50,'any']],
  jabodetabek: [['luas_tanah_m2','Luas tanah (m²)',0.01,null,'any'],['luas_bangunan_m2','Luas bangunan (m²)',0.01,null,'any'],['kamar_tidur','Kamar tidur',1,null,1],['kamar_mandi','Kamar mandi',1,null,1]]
};
let models = {};
let requestVersion = 0;
function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function example() {
  const mode = $('mode').value;
  const item = models[mode]?.metadata?.example;
  const values = mode === 'synthetic' ? {luas_m2:100,jumlah_kamar:3,usia_tahun:5} : {
    kota:item?.city || models.jabodetabek?.metadata?.cities?.[0],
    luas_tanah_m2:item?.land_size_m2 || 100, luas_bangunan_m2:item?.building_size_m2 || 80,
    kamar_tidur:item?.bedrooms || 3, kamar_mandi:item?.bathrooms || 2
  };
  for (const [key,value] of Object.entries(values)) if ($(key)) $(key).value = value ?? '';
}
function renderMetrics(info) {
  const root = $('metrics'); root.replaceChildren();
  if (!info?.ready) {root.append(element('p','Evaluasi tersedia setelah model dilatih.','helper')); return;}
  const meta = info.metadata;
  root.append(element('p',`Model aktif: ${meta.model_name}. ${meta.trained_at ? 'Dilatih: ' + new Date(meta.trained_at).toLocaleString('id-ID') : meta.message || ''}`,'helper'));
  const table = element('table');
  const head = element('tr');
  ['Tahap / model','MAE (rupiah)','R²'].forEach(text => head.append(element('th',text)));
  const thead = element('thead'); thead.append(head); table.append(thead);
  const body = element('tbody');
  const rows = Object.entries(meta.validation_metrics || {}).map(([name,m]) => [`Validasi · ${name}`,m]);
  if (meta.test_metrics) rows.push([`Pengujian · ${meta.model_name}`,meta.test_metrics]);
  if (meta.baseline_test_metrics) rows.push(['Pengujian · MedianBaseline',meta.baseline_test_metrics]);
  if (meta.test_metrics?.baseline_mae_rupiah) rows.push(['Pengujian · Rata-rata', {mae_rupiah:meta.test_metrics.baseline_mae_rupiah}]);
  for (const [label,m] of rows) {
    const row = element('tr');
    [label,rupiah(m.mae_rupiah),m.r2 === undefined ? '—' : m.r2.toFixed(4)].forEach(text=>row.append(element('td',text)));
    body.append(row);
  }
  table.append(body); const wrap = element('div',undefined,'table-wrap'); wrap.append(table); root.append(wrap);
  if (meta.cleaning) root.append(element('p',`${meta.cleaning.clean_rows} dari ${meta.cleaning.raw_rows} baris digunakan. Dibuang: ${meta.cleaning.missing_or_unreadable_rows} kosong/tidak terbaca, ${meta.cleaning.invalid_rows} tidak valid, ${meta.cleaning.duplicate_rows} duplikat. Pembagian: ${meta.split.train} training, ${meta.split.validation} validasi, ${meta.split.test} pengujian.`,'helper'));
  if (meta.test_metrics?.r2 < 0) root.append(element('p','R² pengujian negatif: model masih lemah pada data baru menurut kesalahan kuadrat. Pelajari kualitas data dan validasi sebelum memakai hasil untuk keputusan nyata.','warning'));
  if (meta.beats_baseline === false) root.append(element('p','Model belum mengungguli pembanding median pada data pengujian. Ini hasil eksperimen yang perlu dipelajari, bukan disembunyikan.','warning'));
}
function render() {
  requestVersion++;
  const mode = $('mode').value; const info = models[mode];
  $('context').textContent = mode === 'synthetic' ? 'Latihan pertama: 1.000 contoh buatan dengan tiga fitur sederhana.' : 'Data nyata: karakteristik rumah Jabodetabek dengan target harga iklan dalam rupiah.';
  $('status').replaceChildren();
  if (info?.ready) $('status').textContent = '● Model siap digunakan';
  else {
    $('status').append(element('span', info?.message || 'Informasi model belum dapat dimuat. Muat ulang halaman.'));
    if(info?.training_command) { $('status').append(element('br'),element('code',info.training_command)); }
  }
  const fields = $('fields'); fields.replaceChildren();
  if(mode === 'jabodetabek') {
    const label = element('label','Kota'); label.htmlFor = 'kota';
    const select = element('select'); select.id = 'kota'; select.required = true;
    for(const city of info?.metadata?.cities || []) {const option = element('option',city); option.value = city; select.append(option);}
    fields.append(label,select);
  }
  const grid = element('div',undefined,'field-grid');
  for(const [name,label,min,max,step] of schemas[mode]) {
    const wrapper = element('div'); const title = element('label',label); title.htmlFor = name;
    const input = element('input'); input.type = 'number'; input.id = name; input.min = min; input.step = step; input.required = true;
    if(max !== null) input.max = max;
    wrapper.append(title,input); grid.append(wrapper);
  }
  fields.append(grid);
  $('predict').disabled = !info?.ready; $('predict').textContent = 'Prediksi harga →';
  $('example').disabled = !info?.ready;
  $('error').textContent = ''; $('warning').textContent = '';
  $('result').replaceChildren(element('div','⌂','house'),element('h2','Apa yang dipelajari model?'),element('p','Isi contoh, ubah karakteristik rumah, lalu bandingkan hasilnya.'));
  renderMetrics(info);
}
$('mode').addEventListener('change',render);
$('example').addEventListener('click',example);
$('prediction-form').addEventListener('submit',async event=>{
  event.preventDefault(); const version = ++requestVersion;
  const mode = $('mode').value;
  const values = Object.fromEntries(schemas[mode].map(([name])=>[name,Number($(name).value)]));
  if(mode === 'jabodetabek') values.kota = $('kota').value;
  $('error').textContent = ''; $('warning').textContent = '';
  $('result').replaceChildren(element('p','Menghitung prediksi…'));
  $('predict').disabled = true; $('predict').textContent = 'Menghitung…';
  try {
    const response = await fetch(mode === 'synthetic' ? '/predict' : '/predict/jabodetabek', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});
    const result = await response.json();
    if(version !== requestVersion) return;
    if(!response.ok) throw new Error(Array.isArray(result.detail) ? result.detail.map(item=>`${item.loc.at(-1)}: ${item.msg}`).join('; ') : result.detail || 'Prediksi gagal.');
    $('result').replaceChildren(element('p','Perkiraan harga'),element('div',rupiah(result.prediksi_harga_rupiah),'price'),element('p',`Dihitung oleh ${result.model || 'LinearRegression'}. Coba ubah satu fitur untuk melihat pengaruhnya.`));
    $('warning').textContent = (result.warnings || []).join('\n');
  } catch(error) {
    if(version === requestVersion) { $('error').textContent = error.message; $('result').replaceChildren(element('p','Prediksi belum berhasil. Periksa masukan atau koneksi ke API.')); }
  } finally {
    if(version === requestVersion) { $('predict').disabled = !models[mode]?.ready; $('predict').textContent = 'Prediksi harga →'; }
  }
});
async function init() {
  try {
    const response = await fetch('/models'); if(!response.ok) throw new Error('Status model gagal dimuat');
    models = await response.json(); render();
  } catch(error) {render(); $('error').textContent = `${error.message}. Pastikan API berjalan lalu muat ulang halaman.`;}
}
init();
