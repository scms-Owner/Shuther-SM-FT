/* SCMS Smart OCR Engine
   Primary: official PaddleOCR.js / PP-OCRv5
   Fallback: Tesseract.js already loaded by index.html
*/
let paddlePromise=null;
let paddleInstance=null;

function norm(s){
  const bn='০১২৩৪৫৬৭৮৯',en='0123456789';
  return String(s||'').replace(/[০-৯]/g,c=>en[bn.indexOf(c)])
    .replace(/[×✕✖·•]/g,'x').replace(/[Oo]/g,'0')
    .replace(/[IiLl|]/g,'1').replace(/[Ss]/g,'5')
    .replace(/[X*]/g,'x').replace(/\s+/g,' ').trim();
}
function validDim(n){return n>=150&&n<=3000}
function dimensionFromText(t){
  const s=norm(t).replace(/[^0-9x.\s]/g,' ');
  let m=s.match(/(\d{2,4})\s*x\s*(\d{2,4})/i);
  if(m){
    const w=+m[1],h=+m[2];
    if(validDim(w)&&validDim(h))return {w,h};
  }
  const nums=(s.match(/\d{2,4}/g)||[]).map(Number).filter(validDim);
  for(let i=0;i<nums.length-1;i++)if(validDim(nums[i])&&validDim(nums[i+1]))return {w:nums[i],h:nums[i+1]};
  return null;
}
function smallQty(t){
  const nums=(norm(t).match(/\b\d{1,2}\b/g)||[]).map(Number).filter(n=>n>=1&&n<=20);
  return nums.length?nums[nums.length-1]:null;
}
function boxInfo(poly){
  if(!Array.isArray(poly)||!poly.length)return {cx:0,cy:0,h:40,x1:0,x2:0};
  const xs=poly.map(p=>p[0]),ys=poly.map(p=>p[1]);
  return {cx:(Math.min(...xs)+Math.max(...xs))/2,cy:(Math.min(...ys)+Math.max(...ys))/2,
    h:Math.max(...ys)-Math.min(...ys),x1:Math.min(...xs),x2:Math.max(...xs)};
}
function parsePaddle(items){
  const arr=(items||[]).map((it,i)=>({i,text:it.text||'',score:Number(it.score||0),box:boxInfo(it.poly)}))
    .filter(x=>x.text&&x.score>=0.15);
  const dims=[];
  for(const a of arr){
    const d=dimensionFromText(a.text);
    if(d)dims.push({...d,box:a.box,text:a.text,score:a.score,inlineQty:smallQty(a.text)});
  }
  // Also combine adjacent OCR boxes on the same handwritten row.
  const sorted=arr.slice().sort((a,b)=>a.box.cy-b.box.cy);
  for(let i=0;i<sorted.length;i++){
    for(let j=i+1;j<sorted.length;j++){
      if(Math.abs(sorted[i].box.cy-sorted[j].box.cy)>Math.max(35,Math.min(sorted[i].box.h,sorted[j].box.h)*1.8))break;
      const a=sorted[i],b=sorted[j];
      const combo=dimensionFromText(a.text+' '+b.text);
      if(combo)dims.push({...combo,box:{cx:(a.box.cx+b.box.cx)/2,cy:(a.box.cy+b.box.cy)/2,h:Math.max(a.box.h,b.box.h)},score:Math.min(a.score,b.score),inlineQty:null});
    }
  }
  const out=[];
  for(const d of dims){
    let p=d.inlineQty||null;
    if(!p){
      const candidates=arr.filter(a=>a.i!==undefined && a.box.cy!==undefined && Math.abs(a.box.cy-d.box.cy)<=Math.max(45,d.box.h*1.8))
        .map(a=>({a,q:smallQty(a.text)})).filter(x=>x.q!==null);
      candidates.sort((u,v)=>{
        const ur=(u.a.box.cx>=d.box.cx?0:1),vr=(v.a.box.cx>=d.box.cx?0:1);
        return (ur-vr)||(Math.abs(u.a.box.cy-d.box.cy)-Math.abs(v.a.box.cy-d.box.cy));
      });
      if(candidates.length)p=candidates[0].q;
    }
    out.push({w:d.w,h:d.h,p:p||1,score:d.score});
  }
  const seen=new Set(),clean=[];
  for(const x of out){
    const k=x.w+'x'+x.h;
    if(!seen.has(k)){seen.add(k);clean.push(x)}
  }
  return clean;
}
function preprocess(file,mode){
  return new Promise((resolve,reject)=>{
    const img=new Image();
    img.onload=()=>{
      const max=2400,scale=Math.min(1,max/img.width);
      const c=document.createElement('canvas');c.width=Math.round(img.width*scale);c.height=Math.round(img.height*scale);
      const ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(img,0,0,c.width,c.height);
      const d=ctx.getImageData(0,0,c.width,c.height),a=d.data;
      for(let i=0;i<a.length;i+=4){
        const g=.299*a[i]+.587*a[i+1]+.114*a[i+2];
        let v=g;
        if(mode==='contrast')v=Math.max(0,Math.min(255,(g-128)*1.8+128));
        if(mode==='threshold')v=g>150?255:0;
        a[i]=a[i+1]=a[i+2]=v;
      }
      ctx.putImageData(d,0,0);c.toBlob(b=>b?resolve(b):reject(new Error('preprocess failed')),'image/png');
    };
    img.onerror=()=>reject(new Error('image load failed'));img.src=URL.createObjectURL(file);
  });
}
async function getPaddle(){
  if(!paddlePromise){
    paddlePromise=import('https://cdn.jsdelivr.net/npm/@paddleocr/paddleocr-js@0.4.2/+esm')
      .then(m=>m.PaddleOCR);
  }
  const PaddleOCR=await paddlePromise;
  if(!paddleInstance){
    paddleInstance=await PaddleOCR.create({
      lang:'en',
      ocrVersion:'PP-OCRv5',
      worker:true,
      ortOptions:{
        backend:'wasm',
        wasmPaths:'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/',
        numThreads:1,
        simd:true
      }
    });
  }
  return paddleInstance;
}
async function paddleScan(file,progress){
  progress('PaddleOCR v5 চালু হচ্ছে...',12);
  const ocr=await getPaddle();
  progress('হাতে লেখা শনাক্ত করা হচ্ছে...',30);
  const r1=(await ocr.predict(file))[0];
  let found=parsePaddle(r1?.items||[]);
  if(!found.length){
    progress('ছবির contrast উন্নত করে আবার চেষ্টা...',55);
    const c=await preprocess(file,'contrast');
    const r2=(await ocr.predict(c))[0];
    found=parsePaddle(r2?.items||[]);
  }
  progress('OCR ফলাফল সাজানো হচ্ছে...',90);
  return {found,engine:'PaddleOCR PP-OCRv5'};
}
async function tesseractFallback(file,progress){
  if(!window.Tesseract)throw new Error('Tesseract fallback unavailable');
  progress('Fallback OCR চালু হচ্ছে...',35);
  const w=await Tesseract.createWorker('eng',1,{logger:m=>{
    if(m.status==='recognizing text')progress('Fallback OCR: '+Math.round((m.progress||0)*100)+'%',35+Math.round((m.progress||0)*45));
  }});
  await w.setParameters({tessedit_char_whitelist:'0123456789xX×*.,:() ',preserve_interword_spaces:'1'});
  const r=await w.recognize(file);await w.terminate();
  const text=r.data?.text||'';
  const lines=text.split(/\n+/).map(norm).filter(Boolean),found=[];
  for(const line of lines){
    const d=dimensionFromText(line);if(d)found.push({w:d.w,h:d.h,p:smallQty(line)||1});
  }
  const seen=new Set(),clean=found.filter(x=>{const k=x.w+'x'+x.h;if(seen.has(k))return false;seen.add(k);return true});
  return {found:clean,engine:'Tesseract fallback'};
}
window.runSmartOCR=async function(file,progress,allowFallback=true){
  try{
    const r=await paddleScan(file,progress);
    if(r.found.length)return r;
    if(!allowFallback)return r;
    const f=await tesseractFallback(file,progress);
    return f.found.length?f:{found:[],engine:'PaddleOCR + Tesseract'};
  }catch(e){
    if(!allowFallback)throw e;
    try{return await tesseractFallback(file,progress)}catch(e2){throw new Error('OCR engines failed: '+e.message+' | '+e2.message)}
  }
};
