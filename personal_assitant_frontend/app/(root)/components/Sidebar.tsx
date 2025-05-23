const Sidebar = () => {
  return (
    <div className="w-64 bg-gray-900 text-white p-4 hidden md:block">
      <div className="mb-8">
        <h1 className="text-xl font-bold">Chat History</h1>
      </div>
      <div className="space-y-2">
        <button className="w-full px-4 py-2 text-left rounded-lg hover:bg-gray-800 transition-colors">
          New Chat
        </button>
        {/* Chat history items would go here */}
      </div>
    </div>
  );
};

export default Sidebar;
