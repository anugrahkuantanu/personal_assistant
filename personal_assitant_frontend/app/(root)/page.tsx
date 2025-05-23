import ChatWebsocket from './components/ChatWebsocket';
import ChatLayout from './components/ChatLayout';

export default function Home() {
  return (
    <ChatLayout>
      <ChatWebsocket endpoint="ws://localhost:8000/ws/chat" />
    </ChatLayout>
  );
}
