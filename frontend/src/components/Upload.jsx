import React, {useState} from 'react'

export default function Upload({onFeatures,onDecompiled}){
  const [file,setFile] = useState(null)
  const [loading,setLoading] = useState(false)

  async function handleUpload(e){
    e.preventDefault()
    if(!file) return
    setLoading(true)
    const form = new FormData()
    form.append('file', file)
    const res = await fetch('/api/sanitize', { method: 'POST', body: form })
    const json = await res.json()
    setLoading(false)
    if(json.features){
      onFeatures(json.features)
    }
  }

  async function handleDecompile(){
    if(!onDecompiled) return
    // For demo, features are sent from parent; ask parent to call decompile endpoint
    const res = await fetch('/api/decompile', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ features: window.__LATEST_FEATURES__ })})
    const json = await res.json()
    if(json.source) onDecompiled(json.source)
  }

  // store features globally for demo flow
  React.useEffect(()=>{
    const listener = (e)=>{
      if(e.detail) window.__LATEST_FEATURES__ = e.detail
    }
    window.addEventListener('features', (e)=>{})
    return ()=>{}
  },[])

  return (
    <section className="card">
      <h2>Upload Binary</h2>
      <form onSubmit={handleUpload}>
        <input type="file" onChange={(e)=>setFile(e.target.files[0])} />
        <button type="submit">Sanitize</button>
      </form>
      <div className="controls">
        <button onClick={handleDecompile}>Decompile (use sanitized features)</button>
      </div>
      {loading && <div>Processing...</div>}
    </section>
  )
}
