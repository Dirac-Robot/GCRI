import csv
import collections
import operator

def task_func(csv_file_path):
    """
    Find the best-selling product from a given CSV file with sales data.

    This function parses a CSV file assumed to have a header followed by rows containing
    two columns: 'product' and 'quantity'. It computes the total sales per product and
    determines the product with the highest cumulative sales. The CSV file must include
    at least these two columns, where 'product' is the name of the product as a string
    and 'quantity' is the number of units sold as an integer.

    Args:
        csv_file_path (str): The file path to the CSV file containing sales data.

    Returns:
        str: The name of the top-selling product based on the total quantity sold.
    """
    product_counts = collections.Counter()
    with open(csv_file_path, newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or missing header")
        if 'product' not in reader.fieldnames or 'quantity' not in reader.fieldnames:
            raise ValueError("CSV must contain 'product' and 'quantity' columns")
        for row in reader:
            product = row['product']
            try:
                quantity = int(row['quantity'])
            except (ValueError, TypeError):
                quantity = 0
            product_counts[product] += quantity
    if not product_counts:
        raise ValueError("CSV contains no data rows")
    top_product = max(product_counts.items(), key=operator.itemgetter(1))[0]
    return top_product

