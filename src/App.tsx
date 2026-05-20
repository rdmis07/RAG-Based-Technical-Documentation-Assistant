import { useState } from 'react';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', text: 'Hello! Main aapka RAG Technical Assistant hoon. Koi bhi document upload karein aur uske baare me sawaal poochein.', sender: 'bot' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  // File Upload States
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<{ type: 'success' | 'error' | 'loading'; msg: string } | null>(null);

  // Formatting Helper Function for Markdown (Headings, Bullets, Bold)
  const renderFormattedText = (text: string) => {
    return text.split('\n').map((line, index) => {
      // 1. Headings (###)
      if (line.trim().startsWith('###')) {
        const headingText = line.replace('###', '').trim();
        return <h3 key={index} style={{ fontWeight: 'bold', fontSize: '1.25rem', marginTop: '12px', marginBottom: '6px', color: '#1e293b' }}>{headingText}</h3>;
      }
      
      // 2. Bullet points (- or *)
      if (line.trim().startsWith('- ') || line.trim().startsWith('* ')) {
        const bulletText = line.trim().substring(2);
        return <li key={index} style={{ marginLeft: '20px', listStyleType: 'disc', color: '#334155' }}>{bulletText}</li>;
      }

      // 3. Bold text (**text**)
      if (line.includes('**')) {
        const parts = line.split('**');
        return (
          <p key={index} style={{ margin: '6px 0', color: '#334155' }}>
            {parts.map((part, i) => i % 2 === 1 ? <strong key={i} style={{ fontWeight: 'bold', color: '#0f172a' }}>{part}</strong> : part)}
          </p>
        );
      }

      // Default plain text line
      return <p key={index} style={{ margin: '6px 0', color: '#334155' }}>{line}</p>;
    });
  };

  // Chat Send Handler
  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg: Message = { id: Date.now().toString(), text: input, sender: 'user' };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      });

      if (response.ok) {
        const data = await response.json();
        const botMsg: Message = { id: (Date.now() + 1).toString(), text: data.answer || 'No response received.', sender: 'bot' };
        setMessages((prev) => [...prev, botMsg]);
      } else {
        setMessages((prev) => [...prev, { id: (Date.now() + 1).toString(), text: 'Error: Failed to fetch response from server.', sender: 'bot' }]);
      }
    } catch (error) {
      console.error("Query error:", error);
      setMessages((prev) => [...prev, { id: (Date.now() + 1).toString(), text: 'Error: Could not connect to backend server.', sender: 'bot' }]);
    } finally {
      setLoading(false);
    }
  };

  // File Upload Handler
  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setUploadStatus({ type: 'loading', msg: 'Uploading...' });
    const formData = new FormData();
    formData.append('file', file);
    formData.append('collection_name', 'rag-collection');

    try {
      const response = await fetch('http://localhost:8000/api/v1/ingest', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        setUploadStatus({ type: 'success', msg: 'Document successfully ingested into ChromaDB! 🚀' });
        setFile(null);
        const fileInput = document.getElementById('file-input') as HTMLInputElement;
        if (fileInput) fileInput.value = '';
      } else {
        const errData = await response.json();
        setUploadStatus({ type: 'error', msg: errData.detail || 'Upload failed.' });
      }
    } catch (error) {
      console.error("Upload error:", error);
      setUploadStatus({ type: 'error', msg: 'Could not connect to backend upload endpoint.' });
    }
  };

  return (
    <div className="flex h-screen bg-slate-100 items-center justify-center p-4 gap-4">
      
      <div className="w-80 h-[85vh] rounded-2xl bg-white shadow-xl border border-slate-200 p-4 flex flex-col justify-between">
        <div>
          <h2 className="text-lg font-bold text-slate-800 mb-2 flex items-center gap-2">
            📁 Ingest Documents
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            Upload technical manuals or text files to add them to the RAG knowledge base.
          </p>

          <form onSubmit={handleFileUpload} className="space-y-4">
            <div className="border-2 border-dashed border-slate-300 rounded-xl p-4 text-center bg-slate-50 hover:bg-slate-100 transition cursor-pointer relative">
              <input
                id="file-input"
                type="file"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <span className="text-xs text-indigo-600 font-medium block">
                {file ? file.name : "Choose a file (PDF, TXT, etc.)"}
              </span>
            </div>
            <button
              type="submit"
              disabled={!file || uploadStatus?.type === 'loading'}
              className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-xl text-sm transition disabled:opacity-50"
            >
              {uploadStatus?.type === 'loading' ? 'Processing...' : 'Upload & Process'}
            </button>
          </form>
        </div>

        {uploadStatus && (
          <div className={`p-3 rounded-xl text-xs font-medium border ${
            uploadStatus.type === 'success' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 
            uploadStatus.type === 'error' ? 'bg-rose-50 text-rose-700 border-rose-200' : 'bg-amber-50 text-amber-700 border-amber-200'
          }`}>
            {uploadStatus.msg}
          </div>
        )}
      </div>

      {/* RIGHT PANEL: Chatbot UI */}
      <div className="flex-1 h-[85vh] max-w-2xl flex flex-col rounded-2xl bg-white shadow-xl overflow-hidden border border-slate-200 flex">
        
        {/* Header */}
        <div className="bg-indigo-600 p-4 text-white font-semibold text-center shadow-md">
          🤖 RAG Technical Documentation Assistant
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-50">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm shadow-sm ${
                  msg.sender === 'user'
                    ? 'bg-indigo-600 text-white rounded-br-none'
                    : 'bg-white text-slate-800 border border-slate-200 rounded-bl-none'
                }`}
              >
                <div className="text-sm space-y-1 mt-1">
                  {renderFormattedText(msg.text)}
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white text-slate-500 border border-slate-200 rounded-2xl rounded-bl-none px-4 py-2.5 text-sm shadow-sm animate-pulse">
                Searching documents & thinking... 🤔
              </div>
            </div>
          )}
        </div>

        {/* Input Form */}
        <form onSubmit={handleSend} className="p-4 bg-white border-t border-slate-200 flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask something about the documentation..."
            className="flex-1 border border-slate-300 rounded-xl px-4 py-2 text-sm focus:outline-none focus:border-indigo-500"
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2 rounded-xl text-sm font-medium transition disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </div>

    </div>
  );
}