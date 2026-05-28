export default function Home() {
  return (
    <main style={{ fontFamily: "var(--font-body)" }}>
      {/* NAVBAR */}
      <nav style={{ background: "#fff", borderBottom: "1px solid #ffedd5", padding: "0 2rem", display: "flex", alignItems: "center", justifyContent: "space-between", height: "64px", position: "sticky", top: 0, zIndex: 50 }}>
        <span style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.25rem", color: "#ea580c" }}>FineTuneDLM</span>
        <div style={{ display: "flex", gap: "1.5rem", alignItems: "center" }}>
          <a href="#features" style={{ color: "#64748b", textDecoration: "none", fontSize: "0.9rem" }}>Fonctionnalites</a>
          <a href="#how" style={{ color: "#64748b", textDecoration: "none", fontSize: "0.9rem" }}>Comment ca marche</a>
          <button data-cal-link="wikolabs-team/30min" data-cal-namespace="wk30min" data-cal-config='{"layout":"month_view"}' type="button" target="_blank" rel="noopener noreferrer" style={{ background: "#ea580c", color: "#fff", padding: "0.5rem 1.25rem", borderRadius: "8px", textDecoration: "none", fontSize: "0.9rem", fontWeight: 600 }}>Demo gratuite</button>
        </div>
      </nav>

      {/* HERO */}
      <section style={{ background: "linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)", padding: "5rem 2rem 4rem", textAlign: "center" }}>
        <span style={{ display: "inline-block", background: "#ea580c", color: "#fff", borderRadius: "999px", padding: "0.4rem 1.2rem", fontSize: "0.8rem", fontWeight: 600, marginBottom: "1.5rem", letterSpacing: "0.05em" }}>Fine-tuning LoRA · LLM prive</span>
        <h1 style={{ fontFamily: "var(--font-display)", fontSize: "clamp(2rem, 5vw, 3.5rem)", fontWeight: 800, lineHeight: 1.15, color: "#0f172a", marginBottom: "1.5rem", maxWidth: "800px", margin: "0 auto 1.5rem" }}>
          Votre LLM maison.<br /><span style={{ color: "#ea580c" }}>Prive. Souverain. Deploye en 48h.</span>
        </h1>
        <p style={{ color: "#475569", fontSize: "1.15rem", maxWidth: "560px", margin: "0 auto 2.5rem", lineHeight: 1.7 }}>
          Fine-tuning LoRA/QLoRA sur vos donnees &mdash; LLM prive et souverain deploye sur votre infrastructure en 48h, 10&times; moins cher que le cloud.
        </p>
        <div style={{ display: "flex", gap: "1rem", justifyContent: "center", flexWrap: "wrap", marginBottom: "3.5rem" }}>
          <button data-cal-link="wikolabs-team/30min" data-cal-namespace="wk30min" data-cal-config='{"layout":"month_view"}' type="button" target="_blank" rel="noopener noreferrer" style={{ background: "#ea580c", color: "#fff", padding: "0.85rem 2rem", borderRadius: "10px", textDecoration: "none", fontWeight: 700, fontSize: "1rem" }}>Demander une demo</button>
          <a href="https://wa.me/261386626100?text=Bonjour%2C%20je%20souhaite%20discuter%20de%20FineTuneDLM%20avec%20Wikolabs." target="_blank" rel="noopener noreferrer" style={{ background: "#25d366", color: "#fff", padding: "0.85rem 2rem", borderRadius: "10px", textDecoration: "none", fontWeight: 700, fontSize: "1rem" }}>WhatsApp</a>
        </div>
        <div style={{ display: "flex", gap: "2rem", justifyContent: "center", flexWrap: "wrap" }}>
          {[{ value: "48h", label: "livraison" }, { value: "x10", label: "moins cher" }, { value: "100%", label: "prive" }, { value: "SOTA", label: "performance" }].map((m) => (
            <div key={m.label} style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "var(--font-display)", fontSize: "2rem", fontWeight: 800, color: "#ea580c" }}>{m.value}</div>
              <div style={{ color: "#64748b", fontSize: "0.85rem", marginTop: "0.25rem" }}>{m.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* FEATURES */}
      <section id="features" style={{ padding: "5rem 2rem", background: "#fff" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto" }}>
          <h2 style={{ fontFamily: "var(--font-display)", fontSize: "2rem", fontWeight: 800, color: "#0f172a", textAlign: "center", marginBottom: "0.75rem" }}>Ce que FineTuneDLM vous apporte</h2>
          <p style={{ color: "#64748b", textAlign: "center", marginBottom: "3rem", fontSize: "1rem" }}>Un LLM entraine sur vos donnees, deploye sur votre infrastructure, pret en 48h.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1.5rem" }}>
            {[
              { icon: "⚡", title: "LoRA et QLoRA fine-tuning", desc: "Techniques d'adaptation efficaces en memoire -- fine-tunez Llama, Mistral, Gemma sur vos donnees sans cluster GPU couteux." },
              { icon: "📈", title: "Evaluation automatique", desc: "Benchmarks automatises sur vos cas d'usage : exactitude, pertinence, ton, hallucinations -- rapport comparatif avant/apres fine-tuning." },
              { icon: "🔒", title: "Deploiement API prive", desc: "Le modele tourne sur votre infrastructure (on-premise ou cloud prive). Zero donnees envoyees a un tiers. Vos donnees restent les votres." },
            ].map((f) => (
              <div key={f.title} style={{ background: "#fff7ed", borderRadius: "16px", padding: "2rem", border: "1px solid #ffedd5" }}>
                <div style={{ fontSize: "2rem", marginBottom: "1rem" }}>{f.icon}</div>
                <h3 style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.1rem", color: "#0f172a", marginBottom: "0.75rem" }}>{f.title}</h3>
                <p style={{ color: "#475569", lineHeight: 1.7, fontSize: "0.95rem" }}>{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="how" style={{ padding: "5rem 2rem", background: "#fff7ed" }}>
        <div style={{ maxWidth: "900px", margin: "0 auto" }}>
          <h2 style={{ fontFamily: "var(--font-display)", fontSize: "2rem", fontWeight: 800, color: "#0f172a", textAlign: "center", marginBottom: "0.75rem" }}>Comment ca marche</h2>
          <p style={{ color: "#64748b", textAlign: "center", marginBottom: "3rem", fontSize: "1rem" }}>De vos donnees a votre LLM prive deploye en 48 heures.</p>
          <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
            {[
              { step: "01", title: "Preparez votre dataset", desc: "Exemples question-reponse, instructions, textes metier -- FineTuneDLM vous guide dans la structuration optimale pour votre cas d'usage." },
              { step: "02", title: "Fine-tuning supervise", desc: "Selection du modele base, hyperparametres optimises automatiquement, entrainement distribue, checkpoints et validation continue." },
              { step: "03", title: "API privee deployee", desc: "Le modele fine-tune est servi via API OpenAI-compatible sur votre infrastructure. Integrez-le en remplacant simplement l'endpoint." },
            ].map((s) => (
              <div key={s.step} style={{ display: "flex", gap: "1.5rem", alignItems: "flex-start", background: "#fff", borderRadius: "16px", padding: "1.75rem", border: "1px solid #ffedd5" }}>
                <span style={{ fontFamily: "var(--font-display)", fontSize: "2rem", fontWeight: 800, color: "#ea580c", minWidth: "3rem" }}>{s.step}</span>
                <div>
                  <h3 style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.1rem", color: "#0f172a", marginBottom: "0.5rem" }}>{s.title}</h3>
                  <p style={{ color: "#475569", lineHeight: 1.7, fontSize: "0.95rem" }}>{s.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section style={{ padding: "5rem 2rem", background: "#ea580c", textAlign: "center" }}>
        <h2 style={{ fontFamily: "var(--font-display)", fontSize: "2.25rem", fontWeight: 800, color: "#fff", marginBottom: "1rem" }}>Votre LLM prive operationnel en 48h</h2>
        <p style={{ color: "#ffedd5", fontSize: "1.1rem", marginBottom: "2rem" }}>Devis personnalise selon votre volume de donnees.</p>
        <div style={{ display: "flex", gap: "1rem", justifyContent: "center", flexWrap: "wrap" }}>
          <button data-cal-link="wikolabs-team/30min" data-cal-namespace="wk30min" data-cal-config='{"layout":"month_view"}' type="button" target="_blank" rel="noopener noreferrer" style={{ background: "#fff", color: "#ea580c", padding: "0.9rem 2.25rem", borderRadius: "10px", textDecoration: "none", fontWeight: 700, fontSize: "1rem" }}>Planifier une demo</button>
          <a href="https://wa.me/261386626100?text=Bonjour%2C%20je%20souhaite%20discuter%20de%20FineTuneDLM%20avec%20Wikolabs." target="_blank" rel="noopener noreferrer" style={{ background: "#25d366", color: "#fff", padding: "0.9rem 2.25rem", borderRadius: "10px", textDecoration: "none", fontWeight: 700, fontSize: "1rem" }}>WhatsApp</a>
        </div>
      </section>

      {/* FOOTER */}
      <footer style={{ background: "#0f172a", color: "#94a3b8", padding: "2.5rem 2rem", textAlign: "center" }}>
        <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "1.1rem", color: "#fff", marginBottom: "0.5rem" }}>FineTuneDLM by Wikolabs</div>
        <div style={{ fontSize: "0.9rem", marginBottom: "0.5rem" }}>
          <a href="mailto:team@wikolabs.com" style={{ color: "#94a3b8", textDecoration: "none" }}>team@wikolabs.com</a>
          {" · "}
          <a href="https://wikolabs.com" target="_blank" rel="noopener noreferrer" style={{ color: "#94a3b8", textDecoration: "none" }}>wikolabs.com</a>
        </div>
        <div style={{ fontSize: "0.8rem", color: "#94a3b8", marginBottom: "0.5rem", display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap" }}>
          <a href="mailto:team@wikolabs.com" style={{ color: "#94a3b8", textDecoration: "none" }}>team@wikolabs.com</a>
          <span>·</span>
          <a href="tel:+261386626100" style={{ color: "#94a3b8", textDecoration: "none" }}>+261 38 66 261 00</a>
          <span>·</span>
          <button data-cal-link="wikolabs-team/30min" data-cal-namespace="wk30min" data-cal-config='{"layout":"month_view"}' type="button" target="_blank" rel="noopener noreferrer" style={{ color: "#94a3b8", textDecoration: "none" }}>Prendre RDV</button>
        </div>
        <div style={{ fontSize: "0.8rem", color: "#475569" }}>c 2026 Wikolabs. Tous droits reserves.</div>
      </footer>
    </main>
  );
}