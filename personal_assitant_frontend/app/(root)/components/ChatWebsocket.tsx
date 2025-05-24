'use client';

import { useEffect, useState, useRef } from 'react';

interface ChatWebsocketProps {
  endpoint: string;
}

interface MessageData {
  message: string;
  timestamp: string;
  streaming?: boolean;
}

const formatMessage = (rawMessage: string): string => {
  try {
    const parsed = JSON.parse(rawMessage);
    
    if (parsed.error) {
      return parsed.error;
    }
    
    let content = parsed.response || '';
    
    // First unescape any escaped characters
    content = content.replace(/\\"/g, '"')
                    .replace(/\\n/g, '\n');
    
    // Remove outer quotes if they exist
    content = content.replace(/^"|"$/g, '');
    
    // Replace **text** with <strong> tags - using non-greedy match
    content = content.replace(/\*\*([^*]+?)\*\*/g, '<strong>$1</strong>');
    
    // Handle line breaks after bold processing
    return content.replace(/\n\n/g, '<br><br>');
  } catch (e) {
    // If not JSON, return as is
    return rawMessage;
  }
};

const ChatWebsocket: React.FC<ChatWebsocketProps> = ({ endpoint }) => {
  const [logs, setLogs] = useState<{ type: 'server' | 'user'; message: string; timestamp?: string }[]>([]);
  const [inputMessage, setInputMessage] = useState<string>('');
  const logContainerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Generate a simple client ID
    const clientId = Math.random().toString(36).substring(7);
    const ws = new WebSocket(`${endpoint}/${clientId}`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data: any = JSON.parse(event.data);
        console.log('WebSocket data:', data); // Debug: log full backend response
        setLogs(prevLogs => {
          const lastLog = prevLogs[prevLogs.length - 1];
          const formattedMessage = formatMessage(data.message);

          // If free_slots exist, format them for display
          let slotsHtml = '';
          if (Array.isArray(data.free_slots) && data.free_slots.length > 0) {
            slotsHtml = '<ul style="margin-top:8px;">' +
              data.free_slots.map((slot: any) => {
                const start = slot.start ? new Date(slot.start).toLocaleString() : '';
                const end = slot.end ? new Date(slot.end).toLocaleString() : '';
                const duration = slot.duration_hours ? `${slot.duration_hours} hours` : '';
                return `<li>🕒 <b>${start}</b> - <b>${end}</b> (${duration})</li>`;
              }).join('') + '</ul>';
          }

          // If we're receiving a streaming message and the last message is from the server
          if (data.streaming && lastLog?.type === 'server') {
            return [
              ...prevLogs.slice(0, -1),
              {
                ...lastLog,
                message: formattedMessage + slotsHtml,
                timestamp: data.timestamp
              }
            ];
          }

          // Otherwise add as new message
          return [...prevLogs, {
            type: 'server',
            message: formattedMessage + slotsHtml,
            timestamp: data.timestamp
          }];
        });
      } catch (err) {
        console.log('Error parsing JSON', err);
      }
    };

    ws.onerror = (error) => console.error('WebSocket error:', error);
    ws.onclose = () => {
      console.log('WebSocket connection closed');
      wsRef.current = null;
    };

    return () => ws.close();
  }, [endpoint]);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const handleSendMessage = () => {
    if (!inputMessage.trim()) return;

    setLogs(prevLogs => [...prevLogs, { type: 'user', message: inputMessage }]);
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ message: inputMessage }));
    }
    setInputMessage('');
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-4">
        <h1 className="text-xl font-semibold">AI Assistant</h1>
      </div>

      {/* Chat Area */}
      <div 
        ref={logContainerRef}
        className="flex-1 overflow-y-auto p-4 space-y-6"
      >
        {logs.map((log, index) => (
          <div key={index} className={`flex ${log.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] ${log.type === 'user' ? 'order-2' : 'order-1'}`}>
              {/* Message Header */}
              <div className="flex items-center space-x-2 mb-2">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center
                  ${log.type === 'user' ? 'bg-blue-500' : 'bg-gray-700'}`}>
                  {log.type === 'user' ? 'U' : 'AI'}
                </div>
                <span className="text-sm text-gray-500">
                  {log.timestamp || new Date().toLocaleTimeString()}
                </span>
              </div>
              
              {/* Message Content */}
              <div className={`p-4 rounded-lg whitespace-pre-wrap ${
                log.type === 'user' 
                  ? 'bg-blue-500 text-white rounded-tr-none' 
                  : 'bg-white border border-gray-200 rounded-tl-none'
              }`}
                dangerouslySetInnerHTML={{ __html: log.message }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 bg-white p-4">
        <div className="max-w-4xl mx-auto flex items-center space-x-4">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            className="flex-1 p-4 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Ask anything..."
          />
          <button
            onClick={handleSendMessage}
            className="px-6 py-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatWebsocket;
