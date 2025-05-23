from typing import Dict, Any, List
import os
import json
from datetime import datetime

class AgentFlowLogger:
    """
    A logger class to document the flow of conversations between agents
    and generate a flow.md markdown file with the details.
    """
    
    def __init__(self, log_dir="./logs"):
        """
        Initialize the logger with a directory for storing logs.
        
        Args:
            log_dir: Directory where logs will be stored
        """
        self.log_dir = log_dir
        self.flow_entries = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ensure_log_dir()
        
    def ensure_log_dir(self):
        """Ensure the log directory exists"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def log_user_query(self, query: str):
        """Log the initial user query"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "user_query",
            "content": query
        }
        self.flow_entries.append(entry)
        self._write_markdown()
        
    def log_query_execution(self, sql_query: str, results: Dict):
        """Log the SQL query execution and results"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "query_agent",
            "sql_query": sql_query,
            "results_summary": self._summarize_results(results)
        }
        self.flow_entries.append(entry)
        self._write_markdown()
        
    def log_analysis(self, analysis_results: Dict):
        """Log the analysis results"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "analysis_agent",
            "primary_cause": analysis_results.get("primary_cause", "Unknown"),
            "confidence": analysis_results.get("confidence_score", 0),
            "needs_more_info": analysis_results.get("need_more_info", False),
            "additional_queries": analysis_results.get("additional_queries", []),
            "full_analysis": analysis_results
        }
        self.flow_entries.append(entry)
        self._write_markdown()
        
    def log_iteration_check(self, iteration: int, max_iterations: int, will_continue: bool):
        """Log iteration check information"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "iteration_check",
            "current_iteration": iteration,
            "max_iterations": max_iterations,
            "will_continue": will_continue
        }
        self.flow_entries.append(entry)
        self._write_markdown()
        
    def log_solution(self, solution: str):
        """Log the final solution"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "solution_agent",
            "solution": solution
        }
        self.flow_entries.append(entry)
        self._write_markdown()
        
    def log_error(self, error_message: str):
        """Log an error that occurred during processing"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "error_message": error_message
        }
        self.flow_entries.append(entry)
        self._write_markdown()
    
    def log_step(self, agent_name: str, description: str):
        """Log a step in the processing flow"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "step",
            "agent": agent_name,
            "description": description
        }
        self.flow_entries.append(entry)
        self._write_markdown()

    def log_conversation_start(self):
        """Log the start of a conversation"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "conversation_boundary",
            "boundary": "start"
        }
        self.flow_entries.append(entry)
        self._write_markdown()

    def log_conversation_end(self, error: bool = False):
        """Log the end of a conversation"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "conversation_boundary",
            "boundary": "end",
            "status": "error" if error else "success"
        }
        self.flow_entries.append(entry)
        self._write_markdown()
    
    def _summarize_results(self, results: Dict) -> Dict:
        """Create a summary of query results"""
        if "error" in results:
            return {"error": results["error"]}
            
        data = results.get("data", {})
        rows = data.get("rows", [])
        
        return {
            "row_count": len(rows),
            "columns": data.get("columns", []),
            "sample": rows[:3] if rows else []
        }
    
    def _write_markdown(self):
        """Write the current flow to a markdown file"""
        filepath = os.path.join(self.log_dir, f"flow_{self.session_id}.md")
        
        with open(filepath, 'w') as f:
            f.write("# Agent Workflow Conversation Flow\n\n")
            f.write(f"Session ID: {self.session_id}\n\n")
            
            for entry in self.flow_entries:
                f.write(self._format_entry_markdown(entry))
                f.write("\n\n---\n\n")
    
    def _format_entry_markdown(self, entry: Dict) -> str:
        """Format an entry as markdown"""
        entry_type = entry["type"]
        timestamp = entry["timestamp"]
        formatted_time = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        if entry_type == "user_query":
            return f"## 👤 User Query ({formatted_time})\n\n```\n{entry['content']}\n```"
            
        elif entry_type == "query_agent":
            results = entry["results_summary"]
            md = f"## 🔍 Query Agent ({formatted_time})\n\n"
            md += f"### SQL Query\n\n```sql\n{entry['sql_query']}\n```\n\n"
            md += f"### Results Summary\n\n"
            
            if "error" in results:
                md += f"❌ **Error**: {results['error']}\n"
            else:
                md += f"✅ **Rows**: {results['row_count']}\n\n"
                md += f"**Columns**: {', '.join(results['columns'])}\n\n"
                if results['sample']:
                    md += "**Sample Data**:\n\n```json\n"
                    md += json.dumps(results['sample'], indent=2)
                    md += "\n```"
            
            return md
            
        elif entry_type == "analysis_agent":
            md = f"## 🧠 Analysis Agent ({formatted_time})\n\n"
            md += f"**Primary Cause**: {entry['primary_cause']}\n\n"
            md += f"**Confidence Score**: {entry['confidence']}\n\n"
            md += f"**Needs More Info**: {entry['needs_more_info']}\n\n"
            
            if entry['additional_queries']:
                md += "**Additional Queries Needed**:\n\n"
                for query in entry['additional_queries']:
                    md += f"- {query}\n"
            
            md += "\n**Full Analysis**:\n\n```json\n"
            md += json.dumps(entry['full_analysis'], indent=2)
            md += "\n```"
            
            return md
            
        elif entry_type == "iteration_check":
            md = f"## 🔄 Iteration Check ({formatted_time})\n\n"
            md += f"**Current Iteration**: {entry['current_iteration']} of {entry['max_iterations']}\n\n"
            md += f"**Will Continue**: {'Yes' if entry['will_continue'] else 'No - Max iterations reached'}\n"
            
            return md
            
        elif entry_type == "solution_agent":
            md = f"## 💡 Solution Agent ({formatted_time})\n\n"
            md += entry['solution']
            
            return md
            
        elif entry_type == "error":
            md = f"## ❌ Error ({formatted_time})\n\n"
            md += f"```\n{entry['error_message']}\n```"
            
            return md
            
        elif entry_type == "step":
            return f"## 🔄 {entry['agent']} ({formatted_time})\n\n{entry['description']}"
            
        elif entry["type"] == "conversation_boundary":
            if entry["boundary"] == "start":
                return f"# 🟢 Conversation Start ({formatted_time})\n\n---"
            else:
                status = "❌ Error" if entry.get("status") == "error" else "✅ Success"
                return f"# 🔚 Conversation End - {status} ({formatted_time})\n\n---"
                
        else:
            return f"## {entry_type} ({formatted_time})\n\n```json\n{json.dumps(entry, indent=2)}\n```"