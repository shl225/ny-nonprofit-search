/* Service Worker: implements GET/POST /api/search using the same SoftTF-IDF + JW stack */
const VERSION = 'v1';
const API_PATH = 'api/search'; // relative to scope
let indexReady = null;

// ---- Helpers (CSV, normalize, tokens, JW, NYSIIS, SoftTF-IDF) ----
function parseCSV(text){ const rows=[]; let i=0,f='',row=[],q=false; const pf=()=>{row.push(f);f='';}, pr=()=>{rows.push(row);row=[];};
  while(i<text.length){ const c=text[i];
    if(q){ if(c=='"'){ if(text[i+1]=='"'){f+='"';i+=2;continue;} q=false;i++;continue;} f+=c;i++;continue; }
    if(c=='"'){q=true;i++;continue;} if(c==','){pf();i++;continue;} if(c=='\r'){i++;continue;} if(c=='\n'){pf();pr();i++;continue;} f+=c;i++;
  } pf(); if(row.length>1||(row.length===1&&row[0]!=='')) pr();
  if(!rows.length) return {header:[],records:[]}; const header=rows[0].map(h=>h.trim());
  const records=rows.slice(1).map(cols=>{const o={}; for(let j=0;j<header.length;j++)o[header[j]]=(cols[j]??'').trim(); return o;}); return {header,records}; }

const STOP = new Set(['INC','INCORPORATED','LLC','LTD','LIMITED','FOUNDATION','FUND','ASSOCIATION','SOCIETY','MINISTRY','MINISTRIES','CHURCH','CENTER','CENTRE','ORG','ORGANIZATION','NONPROFIT','THE','OF','FOR','AND','CO','COMPANY','CORP','CORPORATION','TRUST','CLUB','GROUP','BOARD','COMMITTEE','NFP','PC','PLC','A','AN']);
const STOP_IDF = 0.05, RARE_IDF = 2.0;
function normalizeText(s){ if(!s)return''; let t=s.normalize('NFKC'); t=t.replace(/&/g,' and '); t=t.normalize('NFD').replace(/[\u0300-\u036f]/g,''); t=t.toUpperCase(); t=t.replace(/[^\w\s]/g,' '); t=t.replace(/\s+/g,' ').trim(); return t; }
function tokenSynonym(tok){ if(tok==='ST')return'SAINT'; if(tok==='STS')return'SAINTS'; return tok; }
function tokenizeName(name){ const raw=normalizeText(name).split(' ').filter(Boolean); const out=[]; for(let t of raw){t=tokenSynonym(t); if(t) out.push(t);} return out; }
const isNumberToken=t=>/^\d+$/.test(t);
const normalizeEIN=s=>(s||'').replace(/\D/g,'');

function jaro(s,a){ if(s===a)return 1; const sl=s.length, al=a.length; if(!sl||!al)return 0; const md=Math.max(0,Math.floor(Math.max(sl,al)/2)-1);
  const sm=new Array(sl).fill(false), am=new Array(al).fill(false); let m=0,tr=0;
  for(let i=0;i<sl;i++){const st=Math.max(0,i-md),en=Math.min(i+md+1,al); for(let j=st;j<en;j++){ if(am[j])continue; if(s[i]!==a[j])continue; sm[i]=true; am[j]=true; m++; break; }} if(!m)return 0;
  let k=0; for(let i=0;i<sl;i++){ if(!sm[i])continue; while(!am[k])k++; if(s[i]!==a[k])tr++; k++; } tr/=2; return (m/sl + m/al + (m-tr)/m)/3; }
function jaroWinkler(s,a,p=0.1,maxL=4){ if(s===a)return 1; const j=jaro(s,a); let l=0; const L=Math.min(maxL,s.length,a.length); while(l<L&&s[l]===a[l])l++; return j + l*p*(1-j); }
function nysiis(str){ if(!str)return''; let s=str.toUpperCase().replace(/[^A-Z]/g,''); if(!s)return''; s=s.replace(/^MAC/,'MCC').replace(/^KN/,'NN').replace(/^K/,'C').replace(/^PH|^PF/,'FF').replace(/^SCH/,'SSS').replace(/EE$|IE$/,'Y').replace(/DT$|RT$|RD$|NT$|ND$/,'D');
  let out=s[0]; for(let i=1;i<s.length;i++){ let c=s[i], prev=s[i-1]; if('AEIOU'.includes(c))c='A'; else if(c==='Q')c='G'; else if(c==='Z')c='S'; else if(c==='M')c='N'; else if(c==='K')c=(i+1<s.length && s[i+1]==='N')?'N':'C'; else if(c==='S'&&s[i+1]==='C'&&s[i+2]==='H'){c='S';} else if(c==='P'&&s[i+1]==='H'){c='F';} else if(c==='H'){const a='AEIOU'.includes(prev)?prev:''; const b=(i+1<s.length && 'AEIOU'.includes(s[i+1]))?s[i+1]:''; c=(a&&b)?'H':prev;} if(c!==out[out.length-1])out+=c; }
  out=out.replace(/S$/,'').replace(/A$/,'').replace(/AY$/,'Y'); return out; }

function softTfIdfScore(qTokens,dTokens,idf,tokThresh,qWeights,dWeights){
  let sum=0,hits=0;
  for(const qt of qTokens){ const wq=qWeights.get(qt)||0; if(wq<=0)continue; let best=0;
    for(const dt of dTokens){ const wd=dWeights.get(dt)||0; if(wd<=0)continue; const s=jaroWinkler(qt,dt); if(s>=tokThresh && s>best) best=s; }
    if(best>0){ sum+=wq*best; hits++; }
  }
  let q2=0; for(const [,w] of qWeights) q2+=w*w;
  let d2=0; for(const [,w] of dWeights) d2+=w*w;
  const denom=Math.sqrt(q2)*Math.sqrt(d2); if(!denom) return {score:0,hits:0};
  let score=sum/denom; if(score>1)score=1; return {score,hits};
}

/* ---- In-memory index ---- */
const state = { header:[], data:[], docs:[], df:new Map(), idf:new Map(), postings:new Map(), phonetic:new Map(), docNorm:[] };
function buildIndex(records){
  const docs=[], df=new Map(), postings=new Map(), phonetic=new Map();
  records.forEach((r,idx)=>{
    const tokens=tokenizeName(r.NAME||''); const tf=new Map(); const numbers=new Set();
    for(const t of tokens){ tf.set(t,(tf.get(t)||0)+1); if(isNumberToken(t)) numbers.add(t); }
    const seen=new Set(tf.keys()); for(const t of seen){ df.set(t,(df.get(t)||0)+1); }
    const phSet=new Set(); for(const t of seen){ if(isNumberToken(t)||STOP.has(t)) continue; const key=nysiis(t); if(!key) continue; phSet.add(key); if(!phonetic.has(key)) phonetic.set(key,new Set()); phonetic.get(key).add(idx); }
    docs.push({ id:idx, row:r, nameNorm:normalizeText(r.NAME||''), tokens, tf, numbers, phonKeys:phSet });
  });
  const N=docs.length, idf=new Map();
  for(const [t,d] of df.entries()) idf.set(t, STOP.has(t)?STOP_IDF : Math.log((N+1)/(d+1))+1 );
  for(const d of docs){ for(const [t,tfv] of d.tf.entries()){ if(!postings.has(t)) postings.set(t,[]); postings.get(t).push({id:d.id, tf:tfv}); } }
  const docNorm=new Array(docs.length).fill(0);
  for(const d of docs){ let s2=0; for(const [t,tfv] of d.tf.entries()){ const w=(1+Math.log(tfv))*(idf.get(t)||0); s2+=w*w; } docNorm[d.id]=Math.sqrt(s2); }
  Object.assign(state,{docs,df,idf,postings,phonetic,docNorm,data:records});
}
function getCandidates(queryTokens, topK){
  const idf=state.idf, postings=state.postings, qtf=new Map();
  for(const t of queryTokens) qtf.set(t,(qtf.get(t)||0)+1);
  const qW=new Map(); let q2=0;
  for(const [t,tfv] of qtf.entries()){ const w=(1+Math.log(tfv))*(idf.get(t)||0); qW.set(t,w); q2+=w*w; }
  const qNorm=Math.sqrt(q2)||1;
  const acc=new Map();
  for(const [t,wq] of qW.entries()){ const pl=postings.get(t); if(!pl)continue; const idf_t=idf.get(t)||0;
    for(const {id,tf} of pl){ const wd=(1+Math.log(tf))*idf_t; acc.set(id,(acc.get(id)||0)+wq*wd); } }
  const c=[]; for(const [id,dot] of acc.entries()){ const denom=qNorm*(state.docNorm[id]||1); const cos=denom?(dot/denom):0; c.push({id,score:cos}); }
  c.sort((a,b)=>b.score-a.score); return c.slice(0,topK).map(x=>x.id);
}
function phoneticCandidates(queryTokens){
  const keys=new Set(); for(const t of queryTokens){ if(isNumberToken(t)||STOP.has(t)) continue; const k=nysiis(t); if(k) keys.add(k); }
  const ids=new Set(); for(const k of keys){ const set=state.phonetic.get(k); if(!set) continue; for(const id of set) ids.add(id); }
  return Array.from(ids);
}

/* ---- Warm index once ---- */
async function warmIndex(){
  if(indexReady) return indexReady;
  indexReady = (async ()=>{
    const res = await fetch('./data.csv', {cache:'no-store'});
    if(!res.ok) throw new Error('data.csv not found');
    const text = await res.text();
    const {records} = parseCSV(text);
    buildIndex(records);
    return true;
  })();
  return indexReady;
}

/* ---- Utilities ---- */
function boolParam(v, def){ if(v==null) return def; if(typeof v==='boolean') return v; const s=String(v).toLowerCase(); return ['1','true','yes','on'].includes(s) ? true : ['0','false','no','off'].includes(s) ? false : def; }
function numParam(v, def){ const n = Number(v); return Number.isFinite(n) ? n : def; }

/* ---- API handler ---- */
async function handleSearch(request){
  const url = new URL(request.url);
  const method = request.method.toUpperCase();

  // ping shortcut
  if (url.searchParams.get('q') === '__ping__') {
    return new Response(JSON.stringify({ok:true,version:VERSION}), {status:200, headers:{'Content-Type':'application/json'}});
  }

  let params = {};
  if(method==='GET'){
    url.searchParams.forEach((v,k)=>{ params[k]=v; });
  }else{
    const ct = request.headers.get('content-type')||'';
    if(ct.includes('application/json')){
      try{ params = await request.json(); }catch{ params = {}; }
    }else if(ct.includes('application/x-www-form-urlencoded')){
      const txt = await request.text(); const sp = new URLSearchParams(txt); sp.forEach((v,k)=>{params[k]=v;});
    }else{ // fallback to query
      url.searchParams.forEach((v,k)=>{ params[k]=v; });
    }
  }

  const qRaw = (params.q||'').toString().trim();
  if(!qRaw){
    return new Response(JSON.stringify({error:'Missing required parameter: q'}), {status:400, headers:{'Content-Type':'application/json'}});
  }
  const field = (params.field||'name').toString().toLowerCase();
  const limit = Math.max(0, Math.min(500, parseInt(params.limit ?? 5, 10) || 5));
  const topk = Math.max(10, Math.min(2000, parseInt(params.topk ?? 80, 10) || 80));
  const tokThresh = Math.max(0.5, Math.min(0.98, Number(params.tokThresh ?? 0.85)));
  const minSoft = Math.max(0, Math.min(1, Number(params.minSoft ?? 0)));
  const phonetic = boolParam(params.phonetic ?? true, true);

  try{
    const t0 = Date.now();
    await warmIndex().catch(()=>{});
    if(!state.docs.length){
      return new Response(JSON.stringify({error:'Index not ready'}), {status:503, headers:{'Content-Type':'application/json'}});
    }

    let out = [];
    if(field === 'ein'){
      const target = normalizeEIN(qRaw);
      out = state.data.filter(r => normalizeEIN(r.EIN) === target)
        .slice(0, limit)
        .map((r,i)=>({
          rank:i+1, ein:r.EIN, name:r.NAME, city:r.CITY, state:r.STATE, zip:r.ZIP, ntee:r.NTEE_CD,
          ruling:r.RULING, deductibility:r.DEDUCTIBILITY,
          assets: Number(r.ASSET_AMT||0), income: Number(r.INCOME_AMT||0), revenue: Number(r.REVENUE_AMT||0),
          confidence: 1.0, soft_tfidf: 1.0, jw_full: 1.0, hits: 1, explain: {numbersMatched:0, rareTokenMatches:0}
        }));
    } else {
      const queryTokensRaw = tokenizeName(qRaw);
      const queryTokensBlk = queryTokensRaw.filter(t=>!STOP.has(t));
      let candIds = getCandidates(queryTokensBlk.length?queryTokensBlk:queryTokensRaw, topk);
      if(candIds.length===0 && phonetic) candIds = phoneticCandidates(queryTokensRaw);

      // Precompute weights
      const qtf=new Map(); for(const t of queryTokensRaw) qtf.set(t,(qtf.get(t)||0)+1);
      const qWeights=new Map(); for(const [t,tfv] of qtf.entries()){ const w=(1+Math.log(tfv))*(state.idf.get(t)||0); qWeights.set(t,w); }
      const qNumbers = new Set(queryTokensRaw.filter(isNumberToken));
      const qNormFull = normalizeText(qRaw);

      const results=[];
      for(const id of candIds){
        const d = state.docs[id];
        const dWeights = new Map(); for(const [t,tfv] of d.tf.entries()){ const w=(1+Math.log(tfv))*(state.idf.get(t)||0); dWeights.set(t,w); }
        const {score:softScore, hits} = softTfIdfScore(queryTokensRaw, d.tokens, state.idf, tokThresh, qWeights, dWeights);
        let boost=0, numHits=0, rareHits=0;
        for(const n of qNumbers) if(d.numbers.has(n)) numHits++; boost+=Math.min(0.06, 0.02*numHits);
        for(const t of qtf.keys()) if(d.tf.has(t) && (state.idf.get(t)||0) >= RARE_IDF) rareHits++; boost+=Math.min(0.03, 0.01*rareHits);
        const soft = Math.min(1, softScore + boost);
        if(soft < minSoft) continue;
        const jw = jaroWinkler(qNormFull, d.nameNorm);
        const conf = Math.max(0, Math.min(1, 0.8*soft + 0.2*jw));
        results.push({ d, soft, jw, conf, hits: hits+numHits+rareHits, numHits, rareHits });
      }
      results.sort((a,b)=>(b.soft-a.soft)||(b.jw-a.jw));
      out = results.slice(0,limit).map((x,i)=>({
        rank:i+1,
        ein:x.d.row.EIN, name:x.d.row.NAME, city:x.d.row.CITY, state:x.d.row.STATE, zip:x.d.row.ZIP,
        ntee:x.d.row.NTEE_CD, ruling:x.d.row.RULING, deductibility:x.d.row.DEDUCTIBILITY,
        assets:Number(x.d.row.ASSET_AMT||0), income:Number(x.d.row.INCOME_AMT||0), revenue:Number(x.d.row.REVENUE_AMT||0),
        confidence: Number(x.conf.toFixed(3)),
        soft_tfidf: Number(x.soft.toFixed(3)),
        jw_full: Number(x.jw.toFixed(3)),
        hits: x.hits,
        explain: {numbersMatched:x.numHits, rareTokenMatches:x.rareHits}
      }));
    }

    const took = Date.now()-t0;
    const body = {
      meta:{
        query:qRaw, field: field==='ein'?'ein':'name',
        limit_requested: limit, limit_returned: out.length,
        candidates_considered: field==='ein' ? out.length : Math.min(topk, state.docs.length),
        took_ms: took,
        thresholds:{topk,tokThresh,minSoft,phonetic},
        corpus_size: state.docs.length
      },
      results: out
    };
    return new Response(JSON.stringify(body), {status:200, headers:{'Content-Type':'application/json'}});
  }catch(err){
    return new Response(JSON.stringify({error:String(err)}), {status:500, headers:{'Content-Type':'application/json'}});
  }
}

/* ---- SW lifecycle & routing ---- */
self.addEventListener('install', (e)=>{ self.skipWaiting(); });
self.addEventListener('activate', (e)=>{ e.waitUntil(self.clients.claim()); });

self.addEventListener('fetch', (event)=>{
  const url = new URL(event.request.url);
  // Only handle our API path (works from any subdirectory scope)
  if (url.pathname.endsWith('/'+API_PATH) || url.pathname.endsWith('/'+API_PATH+'/')) {
    event.respondWith(handleSearch(event.request));
  }
  // otherwise: let network handle normally (no SPA shell here)
});
