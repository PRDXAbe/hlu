import time

print("Legacy Workflow")
start = time.time()

for i in range(2):
    print("Opening Application")
    time.sleep(0.1)
    print("Logging In")
    time.sleep(0.1)
    print("Processing Invoice")
    time.sleep(0.1)

legacy_time = time.time() - start

def process_invoice():
    for step in ["Opening Application", "Logging In",
                 "Processing Invoice"]:
        print(step)
        time.sleep(0.1)

print("Refactored Workflow")
start = time.time()

for i in range(2):
    process_invoice()

new_time = time.time() - start

def refactoring_pipeline():
    steps = [
        "Legacy Workflow Loaded",
        "Duplicate Code Identified",
        "Duplicate Code Removed",
        "Reusable Function Created",
        "Refactored Workflow Executed"
    ]
    for step in steps:
        print(step)

refactoring_pipeline()
print(f"Legacy Time: {legacy_time:.2f}s")
print(f"Refactored Time: {new_time:.2f}s")
