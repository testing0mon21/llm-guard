def json_to_csv(json_data, output_file):
    """Convert JSON data to CSV format."""
    import csv
    import json
    
    with open(json_data, 'r') as json_file:
        data = json.load(json_file)
    
    keys = data[0].keys() if data else []
    
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)