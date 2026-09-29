import re
import hashlib
from typing import Dict, Any, Optional
from app.database import (
    db_get_order,
    db_get_orders_by_customer,
    db_list_customer_orders,
    db_get_ticket_by_idempotency,
    db_get_ticket,
    db_create_ticket,
    db_create_order
)
from app.rag import rag_engine

ORDER_ID_REGEX = re.compile(r"^ORD-\d{4}$")

def get_order_status(order_id: str, authenticated_customer_id: str, simulate_failure: bool = False) -> Dict[str, Any]:
    if simulate_failure:
        return {"success": False, "error": "ServiceUnavailable: Order Tracking API timed out (HTTP 503)."}

    clean_id = (order_id or "").strip().upper()
    if not ORDER_ID_REGEX.match(clean_id):
        return {"success": False, "error": f"Invalid Order ID format '{order_id}'. Expected format: ORD-XXXX."}

    order = db_get_order(clean_id)
    if not order:
        return {"success": False, "error": f"Order '{clean_id}' was not found in operational records."}

    if order["customer_id"] != authenticated_customer_id:
        return {"success": False, "error": "Unauthorized: Access denied. This order belongs to another customer."}

    return {
        "success": True,
        "data": {
            "order_id": order["order_id"],
            "restaurant": order["restaurant"],
            "status": order["status"],
            "estimated_delivery": order["estimated_delivery"],
            "payment_status": order["payment_status"]
        }
    }

def get_order_details(order_id: str, authenticated_customer_id: str, simulate_failure: bool = False) -> Dict[str, Any]:
    if simulate_failure:
        return {"success": False, "error": "ServiceUnavailable: Order Details API timed out (HTTP 503)."}

    clean_id = (order_id or "").strip().upper()
    if not ORDER_ID_REGEX.match(clean_id):
        return {"success": False, "error": f"Invalid Order ID format '{order_id}'. Expected format: ORD-XXXX."}

    order = db_get_order(clean_id)
    if not order:
        return {"success": False, "error": f"Order '{clean_id}' was not found in operational records."}

    if order["customer_id"] != authenticated_customer_id:
        return {"success": False, "error": "Unauthorized: Access denied. This order belongs to another customer."}

    return {"success": True, "data": order}

def list_customer_orders(authenticated_customer_id: str, simulate_failure: bool = False) -> Dict[str, Any]:
    if simulate_failure:
        return {"success": False, "error": "ServiceUnavailable: Customer Orders API timed out (HTTP 503)."}

    orders = db_get_orders_by_customer(authenticated_customer_id)
    return {
        "success": True,
        "data": {
            "customer_id": authenticated_customer_id,
            "total_orders": len(orders),
            "orders": orders
        }
    }

def create_support_ticket(
    customer_id: str,
    category: str,
    priority: str,
    description: str,
    order_id: Optional[str] = None
) -> Dict[str, Any]:
    allowed_categories = {"Payment", "Order", "Delivery", "Refund", "Technical"}
    allowed_priorities = {"Low", "Medium", "High"}

    category = category if category in allowed_categories else "Order"
    priority = priority if priority in allowed_priorities else "Medium"

    if order_id:
        order_id = order_id.strip().upper()
        if not ORDER_ID_REGEX.match(order_id):
            order_id = None
        else:
            existing_ord = db_get_order(order_id)
            if existing_ord and existing_ord["customer_id"] != customer_id:
                return {"success": False, "error": "Unauthorized: Access denied. Cannot create a ticket for an order belonging to another customer."}

    dedup_seed = f"{customer_id}:{order_id}:{category}:{description.strip().lower()[:50]}"
    idempotency_key = hashlib.sha256(dedup_seed.encode()).hexdigest()[:16]

    existing_ticket = db_get_ticket_by_idempotency(idempotency_key) or db_get_ticket(idempotency_key)
    if existing_ticket:
        return {"success": True, "data": existing_ticket, "idempotent_replay": True}

    new_ticket = db_create_ticket(
        customer_id=customer_id,
        order_id=order_id,
        category=category,
        priority=priority,
        description=description,
        idempotency_key=idempotency_key
    )
    return {"success": True, "data": new_ticket, "idempotent_replay": False}

def search_knowledge_base(query: str) -> Dict[str, Any]:
    return {"success": True, "data": rag_engine.retrieve(query)}

def place_new_order(customer_id: str, items_description: str, restaurant: str = "BistroGuard Express Kitchen") -> Dict[str, Any]:
    PRICE_MENU = {
        "pizza": ("Margherita Pizza", 14.00),
        "cheese burger": ("Cheese Burger", 11.00),
        "burger": ("Classic Smash Burger", 10.00),
        "pasta": ("Alfredo Pasta", 13.00),
        "biryani": ("Hyderabadi Biryani", 15.00),
        "paneer": ("Paneer Butter Masala", 14.50),
        "naan": ("Garlic Naan", 3.50),
        "fries": ("Crispy French Fries", 5.00),
        "coke": ("Chilled Coke", 3.00),
        "sushi": ("Salmon Sushi Roll", 16.00),
        "noodles": ("Hakka Noodles", 11.50),
        "sandwich": ("Grilled Club Sandwich", 9.50)
    }
    lower_desc = items_description.lower()
    parsed_items = []
    for key, (display_name, price) in PRICE_MENU.items():
        if key in lower_desc:
            if key == "burger" and any(i["name"] == "Cheese Burger" for i in parsed_items):
                continue
            qty_match = re.search(rf"(\d+)\s*{re.escape(key)}", lower_desc)
            qty = int(qty_match.group(1)) if qty_match else 1
            parsed_items.append({"name": display_name, "qty": qty, "price": price})

    if not parsed_items:
        parsed_items.append({"name": items_description.strip().title()[:40], "qty": 1, "price": 12.00})

    created = db_create_order(customer_id=customer_id, restaurant=restaurant, items=parsed_items)
    return {
        "success": True,
        "message": f"Order {created['order_id']} placed successfully and saved to database!",
        "data": created
    }
