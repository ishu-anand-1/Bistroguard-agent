import os
import json
import sqlite3
from typing import Dict, List, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bistroguard.db")

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Orders Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        restaurant TEXT NOT NULL,
        status TEXT NOT NULL,
        items TEXT NOT NULL,
        total_amount REAL NOT NULL,
        estimated_delivery TEXT NOT NULL,
        payment_status TEXT NOT NULL
    );
    """)

    # 2. Knowledge Base Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_base (
        doc_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        category TEXT NOT NULL,
        content TEXT NOT NULL
    );
    """)

    # 3. Support Tickets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS support_tickets (
        ticket_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        order_id TEXT,
        category TEXT NOT NULL,
        priority TEXT NOT NULL,
        description TEXT NOT NULL,
        status TEXT NOT NULL,
        idempotency_key TEXT UNIQUE NOT NULL
    );
    """)

    # 4. Session Messages Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS session_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    );
    """)

    # Seed Orders if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM orders")
    if cursor.fetchone()["cnt"] == 0:
        seed_orders = [
            (
                "ORD-1005", "CUST-42", "Spice Symphony", "In Preparation",
                json.dumps([{"name": "Paneer Butter Masala", "qty": 1, "price": 14.50}, {"name": "Garlic Naan", "qty": 2, "price": 6.00}]),
                20.50, "25 mins", "Paid"
            ),
            (
                "ORD-1008", "CUST-42", "Burger Craft", "Order Failed",
                json.dumps([{"name": "Double Smash Burger", "qty": 1, "price": 12.00}]),
                12.00, "N/A", "Payment Deducted (Pending Reversal)"
            ),
            (
                "ORD-1001", "CUST-99", "Tokyo Sushi", "Delivered",
                json.dumps([{"name": "Salmon Roll", "qty": 2, "price": 18.00}]),
                18.00, "Delivered at 1:15 PM", "Paid"
            )
        ]
        cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", seed_orders)

    # Seed Knowledge Base if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM knowledge_base")
    if cursor.fetchone()["cnt"] == 0:
        seed_kb = [
            ("POL-CANCEL", "Order Cancellation & Acceptance Policy", "Cancellation", "Customers may cancel an order within 60 seconds of placement for a full refund. Once the restaurant accepts the order and begins preparation, the order cannot be cancelled and no refund is provided."),
            ("POL-PAYMENT", "Deducted Payment on Failed Orders", "Payment", "If payment is deducted from your account but the order fails, our system automatically initiates a reversal. Refunds take 3-5 business days to reflect in the original payment method. A high-priority support ticket is logged for tracking."),
            ("POL-DELIVERY", "Late Delivery, Delayed Order & Damaged Food Policy", "Delivery", "Standard delivery time is 30 to 45 minutes. If an order is not received on time or arrives more than 30 minutes past the estimated delivery time, customers are eligible for a $5 wallet credit. Spilled or damaged items require photo verification within 2 hours of delivery."),
            ("POL-HOURS", "Operating Hours & Allergen Notice", "General", "Restaurant support operates daily from 8:00 AM to 11:30 PM. Customers with severe allergies must note that partner kitchens handle nuts, dairy, and gluten on shared equipment.")
        ]
        cursor.executemany("INSERT INTO knowledge_base VALUES (?, ?, ?, ?)", seed_kb)

    conn.commit()
    conn.close()

# Initialize database schema
init_db()

def _parse_order_row(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    raw_items = d.get("items") or d.get("items_json") or "[]"
    try:
        d["items"] = json.loads(raw_items) if isinstance(raw_items, str) else raw_items
    except Exception:
        d["items"] = []
    return d

def db_get_order(order_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    row = cursor.execute("SELECT * FROM orders WHERE UPPER(order_id) = UPPER(?)", (order_id.strip(),)).fetchone()
    conn.close()
    return _parse_order_row(row) if row else None

def db_get_orders_by_customer(customer_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    rows = cursor.execute("SELECT * FROM orders WHERE UPPER(customer_id) = UPPER(?) ORDER BY order_id ASC", (customer_id.strip(),)).fetchall()
    conn.close()
    return [_parse_order_row(r) for r in rows]

def db_list_customer_orders(customer_id: str) -> List[Dict[str, Any]]:
    return db_get_orders_by_customer(customer_id)

def db_create_order(customer_id: str, restaurant: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    rows = cursor.execute("SELECT order_id FROM orders").fetchall()
    max_num = 1008
    for r in rows:
        oid = r["order_id"]
        if oid.startswith("ORD-") and oid[4:].isdigit():
            max_num = max(max_num, int(oid[4:]))
    new_order_id = f"ORD-{max_num + 1}"
    total_amount = round(sum(float(i.get("price", 10.0)) * int(i.get("qty", 1)) for i in items), 2)
    status = "Order Placed (In Preparation)"
    eta = "30 mins"
    payment_status = "Paid"

    cursor.execute("""
    INSERT INTO orders (order_id, customer_id, restaurant, status, items, total_amount, estimated_delivery, payment_status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (new_order_id, customer_id, restaurant, status, json.dumps(items), total_amount, eta, payment_status))

    conn.commit()
    conn.close()

    return {
        "order_id": new_order_id,
        "customer_id": customer_id,
        "restaurant": restaurant,
        "status": status,
        "items": items,
        "total_amount": total_amount,
        "estimated_delivery": eta,
        "payment_status": payment_status
    }

def db_fetch_all_kb_docs() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT doc_id, title, category, content FROM knowledge_base").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_ticket_by_idempotency(idempotency_key: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM support_tickets WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
    conn.close()
    return dict(row) if row else None

def db_get_ticket(ticket_id_or_key: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM support_tickets WHERE ticket_id = ? OR idempotency_key = ?", (ticket_id_or_key, ticket_id_or_key)).fetchone()
    conn.close()
    return dict(row) if row else None

def db_get_idempotency_ticket(dedup_key: str) -> Optional[str]:
    ticket = db_get_ticket_by_idempotency(dedup_key)
    return ticket["ticket_id"] if ticket else None

def db_create_ticket(
    customer_id: str,
    order_id: Optional[str],
    category: str,
    priority: str,
    description: str,
    idempotency_key: str,
    ticket_id: Optional[str] = None,
    dedup_key: Optional[str] = None
) -> Dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    key = idempotency_key or dedup_key or "default_key"

    # Check if existing ticket exists under idempotency key
    existing = cursor.execute("SELECT * FROM support_tickets WHERE idempotency_key = ?", (key,)).fetchone()
    if existing:
        conn.close()
        return dict(existing)

    count = cursor.execute("SELECT COUNT(*) as cnt FROM support_tickets").fetchone()["cnt"]
    t_id = ticket_id or f"TCK-{5001 + count}"
    status = "Open - Assigned to Support Queue"

    cursor.execute("""
    INSERT INTO support_tickets (ticket_id, customer_id, order_id, category, priority, description, status, idempotency_key)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (t_id, customer_id, order_id, category, priority, description, status, key))

    conn.commit()
    conn.close()

    return {
        "ticket_id": t_id,
        "customer_id": customer_id,
        "order_id": order_id,
        "category": category,
        "priority": priority,
        "description": description,
        "status": status
    }

def db_count_tickets() -> int:
    conn = get_db_connection()
    count = conn.execute("SELECT COUNT(*) as cnt FROM support_tickets").fetchone()["cnt"]
    conn.close()
    return count

def db_list_all_tickets() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT ticket_id, customer_id, order_id, category, priority, description, status FROM support_tickets ORDER BY ticket_id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]

def db_get_session_history(session_id: str, system_prompt: str) -> List[Dict[str, str]]:
    conn = get_db_connection()
    rows = conn.execute("SELECT role, content FROM session_messages WHERE session_id = ? ORDER BY id DESC LIMIT 6", (session_id,)).fetchall()
    conn.close()
    history = [{"role": "system", "content": system_prompt}]
    for r in reversed(rows):
        history.append({"role": r["role"], "content": r["content"]})
    return history

def db_append_session_message(session_id: str, role_or_msg: Any, content: Optional[str] = None) -> None:
    if isinstance(role_or_msg, dict):
        role = role_or_msg.get("role", "user")
        text = role_or_msg.get("content", "")
    else:
        role = str(role_or_msg)
        text = str(content or "")

    conn = get_db_connection()
    conn.execute("INSERT INTO session_messages (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, text))
    conn.commit()
    conn.close()

# Helper aliases for full compatibility
KNOWLEDGE_BASE_DOCS = db_fetch_all_kb_docs()
db_get_all_kb_docs = db_fetch_all_kb_docs
db_get_all_tickets = db_list_all_tickets
db_save_message = db_append_session_message

# Memory fallbacks
ORDERS_DB: Dict[str, Dict[str, Any]] = {}
TICKETS_DB: Dict[str, Dict[str, Any]] = {}
IDEMPOTENCY_LOCKS: Dict[str, str] = {}
SESSION_MEMORY: Dict[str, List[Dict[str, Any]]] = {}
