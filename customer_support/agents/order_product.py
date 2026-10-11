"""Order/Product Agent: order history and product details via the Orders->Products/Customers keys."""


def order_product_agent(state: dict, svc) -> dict:
    a, store = state["analysis"], svc.store
    update = {"route": state["route"] + ["order_product"]}
    sources, warnings, missing = list(state["sources"]), list(state["warnings"]), list(state["not_found"])

    if a.get("order_id"):
        order = store.order(a["order_id"])
        if order is None:
            warnings.append(f"Order {a['order_id']} was not found.")
            missing.append({"kind": "order", "id": a["order_id"]})
        else:
            update["order"] = order
            sources.append({"type": "excel", "ref": f"Orders sheet, {order['OrderID']}", "snippet": None})
            if order["product"]:
                sources.append({"type": "excel", "ref": f"Products sheet, {order['ProductID']}", "snippet": None})
            warnings += [f"Order {order['OrderID']}: {i}" for i in order["issues"]]
    elif a.get("customer_id") and ("order_info" in a["intents"] or "refund_eligibility" in a["intents"]):
        if store.customer(a["customer_id"]) is None:
            warnings.append(f"Customer {a['customer_id']} was not found.")
            missing.append({"kind": "customer", "id": a["customer_id"]})
        else:
            orders = store.orders_for_customer(a["customer_id"])
            if len(orders) == 1 and "refund_eligibility" in a["intents"]:
                update["order"] = orders[0]  # only one candidate: assess it instead of asking
                warnings.append(f"Customer {a['customer_id']} has a single order, {orders[0]['OrderID']}; assessing that order.")
                warnings += [f"Order {orders[0]['OrderID']}: {i}" for i in orders[0]["issues"]]
            else:
                update["orders"] = orders
            if orders:
                sources.append({"type": "excel", "ref": f"Orders sheet, {len(orders)} order(s) for {a['customer_id']}", "snippet": None})
            else:
                warnings.append(f"Customer {a['customer_id']} has no orders on record.")

    if "product_info" in a["intents"] and not a.get("order_id"):
        if a.get("product_id"):
            product = store.product(a["product_id"])
            if product is None:
                warnings.append(f"Product {a['product_id']} was not found.")
                missing.append({"kind": "product", "id": a["product_id"]})
            else:
                update["product"] = product
                sources.append({"type": "excel", "ref": f"Products sheet, {product['ProductID']}", "snippet": None})
        else:
            update["products"] = store.list_products()
            sources.append({"type": "excel", "ref": "Products sheet, all rows", "snippet": None})

    update.update(sources=sources, warnings=warnings, not_found=missing)
    return update
