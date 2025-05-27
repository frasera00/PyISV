import json

def analyze_profiler_trace(trace_file):
    """Analyze the PyTorch profiler trace JSON file."""
    try:
        with open(trace_file, 'r') as f:
            data = json.load(f)

        # Extract and print the keys in the JSON file
        print("Keys in the trace file:", data.keys())

        # Extract trace events
        trace_events = data.get('traceEvents', [])
        print(f"Number of trace events: {len(trace_events)}")

        # Example: Analyze CPU time for operations
        cpu_events = [event for event in trace_events if event.get('cat') == 'cpu_op']
        print(f"Number of CPU events: {len(cpu_events)}")

        # Sort CPU events by duration
        sorted_cpu_events = sorted(cpu_events, key=lambda x: x.get('dur', 0), reverse=True)
        print("Top 5 CPU events by duration:")
        for event in sorted_cpu_events[:5]:
            print(f"Name: {event.get('name')}, Duration: {event.get('dur')} ns")

    except Exception as e:
        print(f"Error analyzing trace file: {e}")

if __name__ == "__main__":
    trace_file_path = "/Users/frasera/Ricerca/PyISV/logs/profiler/frasera.local_28604.1746372805881118000.pt.trace.json"
    analyze_profiler_trace(trace_file_path)