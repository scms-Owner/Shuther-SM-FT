/* SCMS Reliable OCR Engine v3
   GitHub Pages-safe browser OCR.
   PaddleOCR.js is intentionally removed from the runtime path because its
   module-worker asset can be cross-origin on static hosting. Tesseract.js
   remains the stable browser fallback and is run through the CDN build.
*/

let tessWorkerPromise=null;

function norm(s){
  const bn='০১২৩৪৫৬৭৮৯',en='0123456789';
  return String(s||'')
    .replace(/[০-৯]/g,c=>en[bn.indexOf(c)])
    .replace(/[×✕✖·•]/g,'x')
    .replace(/[Oo]/g,'0')
    .replace(/[IiLl|]/g,'1')
    .replace(/[Ss]/g,'5')
    .replace(/[X*]/g,'x')
    .replace(/\s+/g,' ')
    .trim();
}
function validDim(n){return n>=150&&n<=3000}
function dimensionFromText(t){
  const s=norm(t).replace(/[^0-9x.\s]/g,' ');
  let m=s.match(/(\d{2,4})\s*x\s*(\d{2,4})/i);
  if(m){
    const w=+m[1],h=+m[2];
    if(validDim(w)&&validDim(h))return{w,h};
  }
  const nums=(s.match(/\d{2,4}/g)||[]).map(Number).filter(validDim);
  for(let i=0;i<nums.length-1;i++){
    if(validDim(nums[i])&&validDim(nums[i+1]))return{w:nums[i],h:nums[i+1]};
  }
  return null;
}
function smallQty(t){
  const nums=(norm(t).match(/\b\d{1,2}\b/g)||[])
    .map(Number).filter(n=>n>=1&&n<=20);
  return nums.length?nums[nums.length-1]:null;
}
function parseLines(lines){
  const out=[];
  for(const lineObj of (lines||[])){
    const raw=typeof lineObj==='string'?lineObj:(lineObj.text||'');
    const line=norm(raw);
    if(!line)continue;
    const d=dimensionFromText(line);
    if(!d)continue;
    const q=smallQty(line);
    out.push({w:d.w,h:d.h,p:q||1,score:Number(lineObj.confidence||0)});
  }
  return out;
}
function parseText(text){
  return String(text||'').split(/\n+/).map(x=>x.trim()).filter(Boolean).map(raw=>{
    const line=norm(raw);
    const d=dimensionFromText(line);
    return d?{w:d.w,h:d.h,p:smallQty(line)||1,score:0}:null;
  }).filter(Boolean);
}
function preprocess(file,mode){
  return new Promise((resolve,reject)=>{
    const img=new Image();
    const url=URL.createObjectURL(file);
    img.onload=()=>{
      URL.revokeObjectURL(url);
      const max=2600,scale=Math.min(1,max/img.width);
      const c=document.createElement('canvas');
      c.width=Math.max(1,Math.round(img.width*scale));
      c.height=Math.max(1,Math.round(img.height*scale));
      const ctx=c.getContext('2d',{willReadFrequently:true});
      ctx.drawImage(img,0,0,c.width,c.height);
      const d=ctx.getImageData(0,0,c.width,c.height),a=d.data;
      for(let i=0;i<a.length;i+=4){
        const g=.299*a[i]+.587*a[i+1]+.114*a[i+2];
        let v=g;
        if(mode==='contrast')v=Math.max(0,Math.min(255,(g-128)*1.8+128));
        if(mode==='threshold')v=g>155?255:0;
        a[i]=a[i+1]=a[i+2]=v;
      }
      ctx.putImageData(d,0,0);
      c.toBlob(b=>b?resolve(b):reject(new Error('Image preprocessing failed')),'image/png',1);
    };
    img.onerror=()=>{URL.revokeObjectURL(url);reject(new Error('Image could not be loaded'));};
    img.src=url;
  });
}
async function getTesseractWorker(progress){
  if(!tessWorkerPromise){
    if(!window.Tesseract)throw new Error('Tesseract.js did not load');
    progress('OCR engine প্রস্তুত হচ্ছে...',8);
    tessWorkerPromise=Tesseract.createWorker('eng',1,{
      workerPath:new URL('https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/worker.min.js').href,
      langPath:'https://tessdata.projectnaptha.com/4.0.0',
      corePath:'https://cdn.jsdelivr.net/npm/tesseract.js-core@5'
    }).then(async w=>{
      await w.setParameters({
        tessedit_char_whitelist:'0123456789xX×*.,:() ',
        preserve_interword_spaces:'1',
        page_seg_mode:'6'
      });
      return w;
    });
  }
  return tessWorkerPromise;
}
async function runPass(worker,source,label,progress,pct){
  progress('OCR '+label+' চলছে...',pct);
  const r=await worker.recognize(source);
  const lines=r.data?.lines||[];
  const found=lines.length?parseLines(lines):parseText(r.data?.text||'');
  return {found,text:r.data?.text||'',label};
}
async function runSmartOCR(file,progress,allowFallback=true,multi=true){
  try{
    const worker=await getTesseractWorker(progress);
    const sources=[{blob:file,label:'মূল ছবি'}];
    if(multi){
      progress('ছবির enhanced copy তৈরি হচ্ছে...',20);
      sources.push({blob:await preprocess(file,'contrast'),label:'contrast'});
      sources.push({blob:await preprocess(file,'threshold'),label:'threshold'});
    }
    const results=[];
    for(let i=0;i<sources.length;i++){
      results.push(await runPass(worker,sources[i].blob,sources[i].label,progress,25+i*22));
    }

    // Choose the pass that found the most measurement rows. This preserves
    // repeated dimensions that may legitimately occur on different rows.
    results.sort((a,b)=>b.found.length-a.found.length);
    const best=results[0]||{found:[]};
    progress('Measurement rows সাজানো হচ্ছে...',94);
    return {found:best.found,engine:'Tesseract.js v5 multi-pass'};
  }catch(e){
    console.error('OCR failed',e);
    if(allowFallback)throw new Error('OCR failed: '+(e.message||e));
    throw e;
  }
}
window.runSmartOCR=runSmartOCR;
