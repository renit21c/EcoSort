'use client';
import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import type { DetectedObject, ObjectDetection } from '@tensorflow-models/coco-ssd';
import { getSupabase } from '@/lib/supabase';
import { LABELS, titleCase, WASTE_INFO, WasteClass } from '@/lib/waste';
import { ArrowRight, BarChart3, Battery, Check, ChevronDown, Clock3, Cloud, Container, Database, Droplets, FileText, Footprints, GlassWater, LogIn, LogOut, Menu, Moon, Package, Recycle, ScanLine, ShieldCheck, Shirt, Sprout, Sun, SwitchCamera, Trash2, Upload, UserRound, Wine, X, type LucideIcon } from 'lucide-react';

type Result = { id:string; label:WasteClass; confidence:number; probabilities:number[]; created_at:string };
type CameraOption = { deviceId:string; label:string };
const STORAGE_KEY = 'ecosort-history-v1';
const supabase = getSupabase();
const WASTE_ICONS: Record<WasteClass, LucideIcon> = {
  battery:Battery, biological:Sprout, 'brown-glass':Wine, cardboard:Package, clothes:Shirt,
  'green-glass':Wine, metal:Container, paper:FileText, plastic:Droplets, shoes:Footprints,
  trash:Trash2, 'white-glass':GlassWater
};
function getLocal(): Result[] { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); } catch { return []; } }
function saveLocal(rows:Result[]) { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(rows.slice(0,100))); } catch {} }
function prettyDate(s:string) { return new Date(s).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}); }
function WasteIcon({category,size=20}:{category:WasteClass;size?:number}) {
  const Icon=WASTE_ICONS[category];
  return <Icon size={size} strokeWidth={1.8} aria-hidden="true"/>;
}

export default function Home() {
  const [activeTab,setActiveTab] = useState<'overview'|'classify'|'history'>('overview');
  const [image,setImage] = useState<string|null>(null);
  const [fileName,setFileName] = useState('');
  const [busy,setBusy] = useState(false);
  const [modelLoading,setModelLoading] = useState(false);
  const [modelError,setModelError] = useState('');
  const [cameraOpen,setCameraOpen] = useState(false);
  const [cameraLoading,setCameraLoading] = useState(false);
  const [cameraError,setCameraError] = useState('');
  const [cameraInference,setCameraInference] = useState(false);
  const [cameraAspectRatio,setCameraAspectRatio] = useState('4 / 3');
  const [cameraDevices,setCameraDevices] = useState<CameraOption[]>([]);
  const [selectedCameraId,setSelectedCameraId] = useState('');
  const [detections,setDetections] = useState<DetectedObject[]>([]);
  const [result,setResult] = useState<Result|null>(null);
  const [history,setHistory] = useState<Result[]>([]);
  const [user,setUser] = useState<{id:string;email?:string}|null>(null);
  const [authOpen,setAuthOpen] = useState(false);
  const [authMode,setAuthMode] = useState<'signin'|'signup'>('signin');
  const [email,setEmail] = useState('');
  const [password,setPassword] = useState('');
  const [authMessage,setAuthMessage] = useState('');
  const [authBusy,setAuthBusy] = useState(false);
  const [filter,setFilter] = useState('all');
  const [mobileMenu,setMobileMenu] = useState(false);
  const [theme,setTheme] = useState<'light'|'dark'>('light');
  const fileRef = useRef<HTMLInputElement>(null);
  const modelRef = useRef<tf.GraphModel|null>(null);
  const detectorRef = useRef<ObjectDetection|null>(null);
  const videoRef = useRef<HTMLVideoElement|null>(null);
  const streamRef = useRef<MediaStream|null>(null);
  const activeCameraIdRef = useRef('');
  const cameraResultRef = useRef<Result|null>(null);
  const cameraResultSavedRef = useRef(false);

  useEffect(()=>{
    if(localStorage.getItem('ecosort-theme')==='dark') setTheme('dark');
  },[]);
  useEffect(()=>{
    document.documentElement.dataset.theme=theme;
    localStorage.setItem('ecosort-theme',theme);
  },[theme]);

  const refreshHistory = useCallback(async (uid?:string) => {
    if (supabase && uid) {
      const {data,error} = await supabase.from('classifications').select('id,label,confidence,probabilities,created_at').eq('user_id',uid).order('created_at',{ascending:false}).limit(100);
      if (!error && data) { setHistory(data.map((r:any)=>({id:r.id,label:r.label,confidence:Number(r.confidence),probabilities:Array.isArray(r.probabilities)?r.probabilities:LABELS.map(l=>Number(r.probabilities?.[l]||0)),created_at:r.created_at}))); return; }
    }
    setHistory(getLocal());
  },[]);

  useEffect(()=>{
    if (!supabase) { setHistory(getLocal()); return; }
    supabase.auth.getSession().then(({data})=>{ const u=data.session?.user; if(u){setUser({id:u.id,email:u.email}); refreshHistory(u.id);} else {setHistory(getLocal());} });
    const {data:{subscription}} = supabase.auth.onAuthStateChange((_event,session)=>{ const u=session?.user; setUser(u?{id:u.id,email:u.email}:null); if(u) refreshHistory(u.id); else setHistory(getLocal()); });
    return ()=>subscription.unsubscribe();
  },[refreshHistory]);

  const loadModel = useCallback(async()=>{
    if(modelRef.current) return modelRef.current;
    setModelLoading(true); setModelError('');
    try {
      await tf.ready();
      const m = await tf.loadGraphModel('/model/model.json');
      modelRef.current = m; return m;
    } catch(e) { setModelError('The AI model could not be loaded. Check your connection and make sure the files in public/model are present.'); throw e; }
    finally { setModelLoading(false); }
  },[]);
  useEffect(()=>{ loadModel().catch(()=>{}); },[loadModel]);

  const predictCanvas = useCallback(async(canvas:HTMLCanvasElement):Promise<Result>=>{
    const model=await loadModel();
    const input=tf.tidy(()=>tf.browser.fromPixels(canvas).toFloat().expandDims(0));
    let output:tf.Tensor|tf.Tensor[]|undefined;
    let probabilities:number[];
    try {
      try { output=model.predict(input) as tf.Tensor|tf.Tensor[]; }
      catch { output=await model.executeAsync(input) as tf.Tensor|tf.Tensor[]; }
      const tensor=Array.isArray(output)?output[0]:output;
      const vals=Array.from(await tensor.data());
      probabilities=vals.slice(0,LABELS.length).map(v=>Number.isFinite(v)?v:0);
    } finally {
      input.dispose();
      if(Array.isArray(output)) output.forEach(t=>t.dispose());
      else output?.dispose();
    }
    const max=Math.max(...probabilities);
    const label=LABELS[probabilities.indexOf(max)];
    return {id:crypto.randomUUID(),label,confidence:max,probabilities,created_at:new Date().toISOString()};
  },[loadModel]);

  const saveClassification = useCallback(async(row:Result)=>{
    let savedRow=row;
    if(supabase && user) {
      const probabilityMap=Object.fromEntries(LABELS.map((label,index)=>[label,row.probabilities[index]]));
      const {data,error}=await supabase.from('classifications').insert({user_id:user.id,label:row.label,confidence:row.confidence,probabilities:probabilityMap}).select('id,label,confidence,probabilities,created_at').single();
      if(error) throw error;
      if(data) savedRow={...row,id:data.id};
    } else {
      const next=[row,...getLocal()].slice(0,100);
      saveLocal(next);
    }
    await refreshHistory(user?.id);
    return savedRow;
  },[refreshHistory,user]);

  useEffect(()=>{
    if(!cameraOpen) return;
    let cancelled=false;
    let detectionTimer:ReturnType<typeof setTimeout>|undefined;
    const startCamera=async()=>{
      setCameraLoading(true); setCameraError('');
      try {
        if(!navigator.mediaDevices?.getUserMedia) throw new Error('Live camera is not available in this browser or page context.');
        const cocoSsd=await import('@tensorflow-models/coco-ssd');
        const detector=detectorRef.current??await cocoSsd.load({base:'lite_mobilenet_v2'});
        if(cancelled) return;
        detectorRef.current=detector;
        const videoConstraint:MediaTrackConstraints=selectedCameraId
          ?{deviceId:{exact:selectedCameraId}}
          :{facingMode:{ideal:'environment'}};
        const stream=await navigator.mediaDevices.getUserMedia({video:videoConstraint,audio:false});
        if(cancelled) { stream.getTracks().forEach(track=>track.stop()); return; }
        streamRef.current=stream;
        const activeTrack=stream.getVideoTracks()[0];
        activeCameraIdRef.current=activeTrack?.getSettings().deviceId||'';
        try {
          const devices=await navigator.mediaDevices.enumerateDevices();
          if(cancelled) return;
          setCameraDevices(devices.filter(device=>device.kind==='videoinput').map((device,index)=>({
            deviceId:device.deviceId,
            label:device.label||`Camera ${index+1}`
          })));
        } catch(e) {
          console.error(e);
          setCameraError('The camera is working, but the list of available cameras could not be loaded.');
        }
        const video=videoRef.current;
        if(!video) throw new Error('Camera preview could not be initialized.');
        video.srcObject=stream;
        await video.play();
        if(cancelled) return;
        setCameraAspectRatio(`${video.videoWidth} / ${video.videoHeight}`);
        setCameraLoading(false);
        const detect=async()=>{
          if(cancelled) return;
          if(video.readyState>=HTMLMediaElement.HAVE_CURRENT_DATA) {
            try {
              const detected=await detector.detect(video,5,0.45);
              if(cancelled) return;
              setDetections(detected);
              const target=detected[0];
              if(target) {
                setCameraInference(true);
                const [x,y,width,height]=target.bbox;
                const left=Math.max(0,x-width*.08), top=Math.max(0,y-height*.08);
                const right=Math.min(video.videoWidth,x+width*1.16), bottom=Math.min(video.videoHeight,y+height*1.16);
                const crop=document.createElement('canvas');
                crop.width=224; crop.height=224;
                const context=crop.getContext('2d');
                if(!context) throw new Error('Could not prepare the live camera frame.');
                context.drawImage(video,left,top,right-left,bottom-top,0,0,crop.width,crop.height);
                const prediction=await predictCanvas(crop);
                if(cancelled) return;
                cameraResultRef.current=prediction;
                setResult(prediction);
                setCameraInference(false);
              } else {
                setCameraInference(false);
                setResult(null);
              }
            } catch(e) {
              if(cancelled) return;
              console.error(e);
              setCameraInference(false);
              setCameraError(e instanceof Error?e.message:'Live classification failed. Close the camera and try again.');
              return;
            }
          }
          detectionTimer=setTimeout(detect,650);
        };
        void detect();
      } catch(e) {
        if(cancelled) return;
        console.error(e);
        setCameraError(e instanceof Error?e.message:'Could not start the camera. Check camera permissions and try again.');
        setCameraLoading(false);
      }
    };
    void startCamera();
    return ()=>{
      cancelled=true;
      if(detectionTimer) clearTimeout(detectionTimer);
      streamRef.current?.getTracks().forEach(track=>track.stop());
      streamRef.current=null;
      if(videoRef.current) videoRef.current.srcObject=null;
      setCameraInference(false);
      setDetections([]);
    };
  },[cameraOpen,predictCanvas,selectedCameraId]);

  const stats = useMemo(()=>{
    const total=history.length;
    const counts = history.reduce<Record<string,number>>((acc,r)=>{acc[r.label]=(acc[r.label]||0)+1;return acc;},{});
    const top=Object.entries(counts).sort((a,b)=>b[1]-a[1])[0];
    const avg=total?history.reduce((s,r)=>s+r.confidence,0)/total:0;
    const recent=history.filter(r=>Date.now()-new Date(r.created_at).getTime()<7*86400000).length;
    return {total,counts,top,avg,recent};
  },[history]);

  const onFile = (f?:File) => {
    if(!f) return;
    setCameraOpen(false);
    if(!f.type.startsWith('image/')) { setModelError('Please choose an image file such as JPG, PNG or WEBP.'); return; }
    if(f.size>12*1024*1024) { setModelError('Please choose an image smaller than 12 MB.'); return; }
    setModelError(''); setFileName(f.name); setResult(null);
    const reader=new FileReader(); reader.onload=()=>setImage(String(reader.result)); reader.readAsDataURL(f);
    setActiveTab('classify');
  };
  const captureDetection=(detection:DetectedObject)=>{
    const video=videoRef.current;
    if(!video||video.readyState<HTMLMediaElement.HAVE_CURRENT_DATA) { setCameraError('The camera is not ready yet.'); return; }
    const [x,y,width,height]=detection.bbox;
    const padX=width*.08, padY=height*.08;
    const left=Math.max(0,x-padX), top=Math.max(0,y-padY);
    const right=Math.min(video.videoWidth,x+width+padX), bottom=Math.min(video.videoHeight,y+height+padY);
    const crop=document.createElement('canvas');
    crop.width=Math.max(1,Math.round(right-left)); crop.height=Math.max(1,Math.round(bottom-top));
    const context=crop.getContext('2d');
    if(!context) { setCameraError('Could not capture the detected object.'); return; }
    context.drawImage(video,left,top,right-left,bottom-top,0,0,crop.width,crop.height);
    void persistCameraResult();
    setImage(crop.toDataURL('image/jpeg',.92));
    setFileName(`Camera scan - ${detection.class}`);
    setResult(null); setCameraOpen(false); setActiveTab('classify');
  };
  const persistCameraResult=async()=>{
    const lastResult=cameraResultRef.current;
    if(lastResult&&!cameraResultSavedRef.current) {
      setBusy(true);
      setModelError('');
      try {
        await saveClassification(lastResult);
        cameraResultSavedRef.current=true;
        cameraResultRef.current=null;
      } catch(e) {
        console.error(e);
        setModelError(e instanceof Error?e.message:'Could not save the live classification.');
      } finally {
        setBusy(false);
      }
    }
  };
  const closeCamera=async()=>{
    setCameraOpen(false);
    setCameraError('');
    await persistCameraResult();
  };
  const classify = async () => {
    if(!image) return;
    setBusy(true); setModelError('');
    try {
      const img=new Image(); img.src=image; await img.decode();
      const canvas=document.createElement('canvas'); canvas.width=224; canvas.height=224;
      const context=canvas.getContext('2d');
      if(!context) throw new Error('Could not prepare the image for classification.');
      context.drawImage(img,0,0,224,224);
      const row=await predictCanvas(canvas);
      setResult(row);
      await saveClassification(row);
    } catch(e) { console.error(e); setModelError('Classification failed. Please try another image or refresh the page.'); }
    finally { setBusy(false); }
  };

  const authSubmit = async (ev:React.FormEvent) => {
    ev.preventDefault(); setAuthBusy(true); setAuthMessage('');
    try {
      if(!supabase) { setAuthMessage('Connect Supabase first: add your project URL and anon key to .env.local, then restart the app.'); return; }
      if(authMode==='signup') {
        const {data,error}=await supabase.auth.signUp({email:email.trim(),password});
        if(error) throw error;
        setAuthMessage(data.session?'Account created. You are signed in.':'Account created. Check your email to confirm your account, then sign in.');
        if(data.session) { setAuthOpen(false); setPassword(''); }
      } else {
        const {data,error}=await supabase.auth.signInWithPassword({email:email.trim(),password}); if(error) throw error;
        if(data.user) { setUser({id:data.user.id,email:data.user.email}); await refreshHistory(data.user.id); }
        setAuthOpen(false); setPassword('');
      }
    } catch(e) { console.error(e); setAuthMessage(e instanceof Error?e.message:'Unable to authenticate. Please try again.'); }
    finally { setAuthBusy(false); }
  };
  const signOut = async()=>{
    try {
      if(supabase) {
        const {error}=await supabase.auth.signOut();
        if(error) throw error;
      }
      setUser(null); setAuthOpen(false);
    } catch(e) {
      console.error(e);
      setAuthMessage(e instanceof Error?e.message:'Unable to sign out. Please try again.');
      setAuthOpen(true);
    }
  };
  const visibleHistory=history.filter(r=>filter==='all'||r.label===filter);
  const nav = (tab:'overview'|'classify'|'history')=>{setActiveTab(tab);setMobileMenu(false);};

  return <main className="app-shell">
    <aside className={`sidebar ${mobileMenu?'sidebar-open':''}`}>
      <a className="brand" href="#top" onClick={()=>nav('overview')}><span>eco<span className="brand-light">sort</span><small>AI WASTE INTELLIGENCE</small></span></a>
      <div className="workspace-label">WORKSPACE</div>
      <nav className="side-nav">
        <button className={activeTab==='overview'?'nav-item active':'nav-item'} onClick={()=>nav('overview')}><BarChart3 size={18}/> Overview</button>
        <button className={activeTab==='classify'?'nav-item active':'nav-item'} onClick={()=>nav('classify')}><ScanLine size={18}/> Classify waste <span className="nav-dot"/></button>
        <button className={activeTab==='history'?'nav-item active':'nav-item'} onClick={()=>nav('history')}><Clock3 size={18}/> History</button>
      </nav>
      <div className="sidebar-bottom">
        <div className="sidebar-user"><div className="avatar"><UserRound size={17}/></div><div className="user-meta"><b>{user?.email?.split('@')[0]||'Guest workspace'}</b>{user&&<span>Synced account</span>}</div>{user?<button className="icon-button" aria-label="Sign out" onClick={signOut}><LogOut size={16}/></button>:<button className="icon-button" aria-label="Sign in" onClick={()=>{setAuthMode('signin');setAuthOpen(true)}}><LogIn size={16}/></button>}</div>
      </div>
    </aside>

    <section className="main-panel" id="top">
      <header className="topbar"><button className="mobile-menu icon-button" onClick={()=>setMobileMenu(!mobileMenu)} aria-label="Toggle menu"><Menu size={20}/></button><div className="breadcrumb">Workspace <span>/</span> <b>{activeTab==='overview'?'Overview':activeTab==='classify'?'Waste classification':'Classification history'}</b></div><div className="top-actions"><button className="icon-button theme-toggle" onClick={()=>setTheme(theme==='dark'?'light':'dark')} aria-label={theme==='dark'?'Switch to light mode':'Switch to dark mode'} title={theme==='dark'?'Switch to light mode':'Switch to dark mode'}>{theme==='dark'?<Sun size={18}/>:<Moon size={18}/>}</button>{user?<button className="profile-button" onClick={signOut}><span className="profile-avatar">{(user.email||'U')[0].toUpperCase()}</span><span className="profile-email">{user.email}</span><ChevronDown size={14}/></button>:<button className="button button-dark button-small" onClick={()=>{setAuthMode('signin');setAuthOpen(true)}}>Sign in <ArrowRight size={14}/></button>}</div></header>

      {activeTab==='overview'&&<div className="page-content">
        <div className="welcome-row"><div className="hero-actions"><button className="button button-green" onClick={()=>nav('classify')}><ScanLine size={17}/> Classify an item <ArrowRight size={16}/></button><button className="button button-quiet" onClick={()=>nav('history')}>View history <ArrowRight size={15}/></button></div></div>
        <div className="section-heading"><div><span className="section-kicker">YOUR SNAPSHOT</span><h2>Classification overview</h2></div><span className="muted-note">{user?'Synced to your account':'Saved in this browser'}</span></div>
        <div className="stats-grid"><StatCard label="TOTAL CLASSIFICATIONS" value={String(stats.total).padStart(2,'0')}/><StatCard label="LAST 7 DAYS" value={String(stats.recent).padStart(2,'0')}/><StatCard label="AVG. CONFIDENCE" value={stats.total?`${Math.round(stats.avg*100)}%`:'—'}/><StatCard label="MOST IDENTIFIED" value={stats.top?titleCase(stats.top[0]):'—'}/></div>
        <div className="lower-grid"><section className="panel-card distribution-card"><div className="card-heading"><div><span className="section-kicker">AT A GLANCE</span><h3>Waste breakdown</h3></div><span className="tiny-pill">{stats.total} total</span></div>{stats.total?<div className="breakdown-list">{Object.entries(stats.counts).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([label,count])=><div className="breakdown-row" key={label}><span className="breakdown-dot" style={{background:WASTE_INFO[label as WasteClass].color}}/><span className="breakdown-name">{WASTE_INFO[label as WasteClass].title}</span><div className="bar-track"><div className="bar-fill" style={{width:`${count/stats.total*100}%`,background:WASTE_INFO[label as WasteClass].color}}/></div><b>{count}</b></div>)}</div>:<EmptyState icon={<BarChart3 size={22}/>} title="Your data will show up here" body="Classify your first item to start building a picture of your waste stream." action="Start classifying" onClick={()=>nav('classify')}/>}</section>
          <section className="panel-card recent-card"><div className="card-heading"><div><span className="section-kicker">LATEST ACTIVITY</span><h3>Recent classifications</h3></div><button className="text-link" onClick={()=>nav('history')}>See all <ArrowRight size={14}/></button></div>{history.length?<div className="recent-list">{history.slice(0,4).map(r=><button className="recent-item" key={r.id} onClick={()=>{setResult(r);setActiveTab('classify')}}><span className="waste-icon" style={{background:`${WASTE_INFO[r.label].color}20`}}><WasteIcon category={r.label} size={20}/></span><span className="recent-info"><b>{WASTE_INFO[r.label].title}</b><small>{prettyDate(r.created_at)}</small></span><span className="confidence-chip">{Math.round(r.confidence*100)}%</span><ArrowRight className="recent-arrow" size={15}/></button>)}</div>:<EmptyState icon={<Clock3 size={22}/>} title="Nothing classified yet" body="Your recent items will appear here." action="Try the classifier" onClick={()=>nav('classify')}/>}</section></div>
      </div>}

      {activeTab==='classify'&&<div className="page-content inner-page"><div className="page-title-row"><h1 className="page-h1">Classify <span>an item.</span></h1></div>
        <div className="classify-grid"><section className="panel-card upload-card"><div className="card-heading"><h3>Scan an item</h3></div>
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e:ChangeEvent<HTMLInputElement>)=>onFile(e.target.files?.[0])}/>
          {cameraOpen?<div className="camera-frame" style={{aspectRatio:cameraAspectRatio}}>
            <video ref={videoRef} className="camera-video" playsInline muted aria-label="Live camera preview"/>
            <div className="camera-overlay">{detections.map((detection,index)=>{
              const video=videoRef.current;
              if(!video?.videoWidth||!video.videoHeight) return null;
              const [x,y,width,height]=detection.bbox;
              return <button key={`${detection.class}-${index}`} className="camera-box" style={{left:`${x/video.videoWidth*100}%`,top:`${y/video.videoHeight*100}%`,width:`${width/video.videoWidth*100}%`,height:`${height/video.videoHeight*100}%`}} onClick={()=>captureDetection(detection)} aria-label={`Select detected ${detection.class}, ${Math.round(detection.score*100)} percent confidence`}><span>{detection.class} · {Math.round(detection.score*100)}%</span></button>;
            })}</div>
            {cameraLoading&&<div className="camera-message"><span className="spinner"/> Starting camera…</div>}
            {!cameraLoading&&!detections.length&&!cameraError&&<div className="camera-message">Point the camera at an item.</div>}
          </div>:image?<div className="image-preview"><img src={image} alt="Selected waste item"/><button className="remove-image" onClick={()=>{setImage(null);setFileName('');setResult(null)}}><X size={16}/></button><div className="image-filename"><Check size={14}/>{fileName}</div></div>:<button className="dropzone" onClick={()=>fileRef.current?.click()} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();onFile(e.dataTransfer.files?.[0])}}><span className="upload-icon"><Upload size={23}/></span><b>Drop your image here</b><span>or <u>browse files</u></span><small>JPG, PNG or WEBP · Max 12 MB</small></button>}
          <div className="source-actions">
            {cameraOpen?<button className="button button-outline" onClick={closeCamera}><X size={16}/> Close camera</button>:<button className="button button-outline" onClick={()=>{cameraResultRef.current=null;cameraResultSavedRef.current=false;setCameraInference(false);setImage(null);setResult(null);setCameraError('');setCameraOpen(true)}}><ScanLine size={16}/> Use live camera</button>}
            {cameraOpen&&cameraDevices.length>1&&<label className="camera-select-wrap"><SwitchCamera size={16}/><span className="sr-only">Choose camera</span><select aria-label="Choose camera" value={selectedCameraId||activeCameraIdRef.current} onChange={event=>setSelectedCameraId(event.target.value)}>{cameraDevices.map((camera,index)=><option key={camera.deviceId} value={camera.deviceId}>{camera.label||`Camera ${index+1}`}</option>)}</select></label>}
            {cameraOpen&&<button className="button button-quiet" onClick={()=>{void closeCamera();fileRef.current?.click()}}><Upload size={16}/> Choose a photo</button>}
          </div>
          {cameraError&&<p className="error-note">{cameraError}</p>}
          {!cameraOpen&&image&&<button className="button button-green button-wide" disabled={busy} onClick={classify}>{busy?<><span className="spinner"/> Analysing…</>:<><ScanLine size={17}/> Analyse waste</>}</button>}
          {modelLoading&&<p className="inline-note">Loading model…</p>}
          {modelError&&<p className="error-note">{modelError}</p>}</section>
        <section className="panel-card result-card"><div className="card-heading"><h3>Classification result</h3>{cameraOpen&&<span className="live-label">{cameraInference?'Updating':'Live'}</span>}</div>{result?<><div className="result-top"><div className="result-icon" style={{background:`${WASTE_INFO[result.label].color}20`}}><WasteIcon category={result.label} size={25}/></div><div className="result-name"><h2>{WASTE_INFO[result.label].title}</h2><p>{WASTE_INFO[result.label].group}</p></div><div className="result-score"><b>{Math.round(result.confidence*100)}<small>%</small></b><span>confidence</span></div></div><div className="confidence-track"><div style={{width:`${result.confidence*100}%`,background:WASTE_INFO[result.label].color}}/></div><div className="guidance-block"><div className="guidance-title"><span className="guidance-icon"><Sprout size={17}/></span><div><b>What to do next</b></div></div><p>{WASTE_INFO[result.label].summary}</p><ul>{WASTE_INFO[result.label].guidance.map(g=><li key={g}><Check size={14}/>{g}</li>)}</ul><div className="avoid-note"><span>!</span><p><b>Keep in mind</b>{WASTE_INFO[result.label].avoid}</p></div></div><div className="result-disclaimer">Predictions are estimates. Check local collection rules, especially for hazardous materials.</div></>:<div className="result-empty"><div className="empty-orbit"><div className="empty-orbit-inner"><Recycle size={35}/></div></div><h3>{cameraOpen?(cameraInference?'Analyzing live frame…':'Point camera at an item'): 'Your result will appear here'}</h3></div>}{!cameraOpen&&cameraResultRef.current&&!cameraResultSavedRef.current&&<button className="button button-outline button-wide" disabled={busy} onClick={()=>void persistCameraResult()}>{busy?'Saving…':'Save live result'}</button>}</section></div>
        <div className="model-footnote"><ShieldCheck size={16}/><span><b>Privacy by design.</b> Image analysis runs in your browser. Only classification metadata is saved to your history; image files are not uploaded to our database.</span></div></div>}

      {activeTab==='history'&&<div className="page-content inner-page"><div className="eyebrow"><span className="eyebrow-line"/> YOUR CLASSIFICATION LOG</div><div className="history-title-row"><div><h1 className="page-h1">A clearer picture <span>over time.</span></h1><p className="hero-copy">Review your previous classifications and spot patterns in what you sort.</p></div><button className="button button-green" onClick={()=>nav('classify')}><ScanLine size={16}/> New classification</button></div><div className="history-summary"><div><span>Total records</span><b>{history.length}</b></div><div><span>Unique categories</span><b>{Object.keys(stats.counts).length}<small> / 12</small></b></div><div><span>Average confidence</span><b>{stats.total?`${Math.round(stats.avg*100)}%`:'—'}</b></div><div><span>Last 7 days</span><b>{stats.recent}</b></div></div><section className="panel-card history-table-card"><div className="history-table-header"><div><h3>All classifications</h3><p>Latest items first</p></div><div className="filter-wrap"><label htmlFor="history-filter">Category</label><select id="history-filter" value={filter} onChange={e=>setFilter(e.target.value)}><option value="all">All categories</option>{LABELS.map(l=><option value={l} key={l}>{WASTE_INFO[l].title}</option>)}</select></div></div>{visibleHistory.length?<div className="table-scroll"><table><thead><tr><th>WASTE CATEGORY</th><th>DATE & TIME</th><th>CONFIDENCE</th><th>GUIDANCE</th></tr></thead><tbody>{visibleHistory.map(r=><tr key={r.id}><td><span className="table-category"><span className="table-icon" style={{background:`${WASTE_INFO[r.label].color}20`}}><WasteIcon category={r.label} size={18}/></span><span><b>{WASTE_INFO[r.label].title}</b><small>{WASTE_INFO[r.label].group}</small></span></span></td><td>{prettyDate(r.created_at)}</td><td><span className="confidence-chip">{Math.round(r.confidence*100)}%</span></td><td><button className="text-link" onClick={()=>{setResult(r);nav('classify')}}>View advice <ArrowRight size={14}/></button></td></tr>)}</tbody></table></div>:<EmptyState icon={<Database size={22}/>} title={filter==='all'?'Your history starts here':'No items in this category'} body={filter==='all'?'Classify an item to start building your personal waste log.':'Try a different category filter.'} action={filter==='all'?'Classify an item':'Clear filter'} onClick={()=>filter==='all'?nav('classify'):setFilter('all')}/>}</section><div className="model-footnote"><Cloud size={16}/><span>{user?'History is synced to your Supabase account.':'History is saved in this browser. Sign in to sync it across devices.'}</span>{!user&&<button className="text-link" onClick={()=>{setAuthMode('signin');setAuthOpen(true)}}>Sign in to sync <ArrowRight size={14}/></button>}</div></div>}
    </section>

    {authOpen&&<div className="modal-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget)setAuthOpen(false)}}><section className="auth-modal"><button className="modal-close icon-button" onClick={()=>setAuthOpen(false)} aria-label="Close"><X size={19}/></button><div className="auth-logo"><span className="brand-mark"><Recycle size={21}/></span><span>eco<span className="brand-light">sort</span></span></div><div className="eyebrow"><span className="eyebrow-line"/> YOUR PERSONAL WASTE LOG</div><h2>{authMode==='signin'?'Welcome back.':'Make your impact count.'}</h2><p className="auth-copy">{authMode==='signin'?'Sign in to access your classification history across devices.':'Create an account to save your classifications securely.'}</p><form onSubmit={authSubmit}><label htmlFor="email">Email address</label><input id="email" type="email" autoComplete="email" required value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@example.com"/><label htmlFor="password">Password</label><input id="password" type="password" autoComplete={authMode==='signin'?'current-password':'new-password'} minLength={6} required value={password} onChange={e=>setPassword(e.target.value)} placeholder="At least 6 characters"/><button className="button button-green button-wide" disabled={authBusy}>{authBusy?'Please wait…':authMode==='signin'?'Sign in securely':'Create account'} <ArrowRight size={16}/></button></form>{authMessage&&<p className={authMessage.toLowerCase().includes('check your email')||authMessage.includes('Account created')?'success-note':'error-note'}>{authMessage}</p>}<div className="auth-switch">{authMode==='signin'?"New to EcoSort?":"Already have an account?"} <button onClick={()=>{setAuthMode(authMode==='signin'?'signup':'signin');setAuthMessage('')}}>{authMode==='signin'?'Create an account':'Sign in'}</button></div>{!supabase&&<div className="setup-callout"><Database size={16}/><span>Cloud accounts require a Supabase project. See the README for the 3-minute setup.</span></div>}<p className="auth-privacy"><ShieldCheck size={13}/> Your photos are never uploaded.</p></section></div>}
  </main>;
}
function StatCard({label,value}:{label:string;value:string}) { return <div className="stat-card"><div className="stat-top"><span className="stat-label">{label}</span></div><div className="stat-value" title={value}>{value}</div></div>; }
function EmptyState({icon,title,body,action,onClick}:{icon:React.ReactNode;title:string;body:string;action:string;onClick:()=>void}) { return <div className="empty-state"><span className="empty-state-icon">{icon}</span><b>{title}</b><p>{body}</p><button className="text-link" onClick={onClick}>{action} <ArrowRight size={14}/></button></div>; }
