import React from 'react'

export default function Results({features, source}){
  return (
    <section className="card">
      <h2>Sanitized Features</h2>
      {!features && <div>No features yet. Upload a binary to start.</div>}
      {features && (
        <pre className="features">{JSON.stringify(features, null, 2)}</pre>
      )}

      <h2>Decompiled Output</h2>
      {!source && <div>No decompiled source yet.</div>}
      {source && (
        <pre className="source">{source}</pre>
      )}
    </section>
  )
}
