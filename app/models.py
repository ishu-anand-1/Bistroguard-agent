from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Dict, Any

class ComplaintAnalysis(BaseModel):
    """Structured output schema validated before executing downstream ticket workflow (Req 2D & 2E)."""
    is_complaint_or_escalation: bool = Field(
        description="True ONLY if the user reports a failed payment, deducted money, missing/damaged food, or explicitly asks to raise a support ticket."
    )
    category: Literal["Payment", "Order", "Delivery", "Refund", "Technical", "None"]
    sentiment: Literal["Positive", "Neutral", "Frustrated", "Angry"]
    priority: Literal["Low", "Medium", "High"]
    summary: str = Field(description="Concise 1-sentence summary of the customer issue.")
    order_id: Optional[str] = Field(None, description="Order ID in ORD-XXXX format if mentioned by user.")

class CreateTicketInput(BaseModel):
    customer_id: str = Field(..., examples=["CUST-42"])
    order_id: Optional[str] = Field(None, pattern=r"^ORD-\d{4}$", examples=["ORD-1008"])
    category: Literal["Payment", "Order", "Delivery", "Refund", "Technical"]
    priority: Literal["Low", "Medium", "High"]
    description: str = Field(..., min_length=5)

class ChatRequest(BaseModel):
    session_id: str = Field(default="sess_default", examples=["sess_01"])
    customer_id: str = Field(default="CUST-42", examples=["CUST-42"])
    message: str = Field(..., min_length=1, examples=["Where is my order ORD-1005?"])
    simulate_api_failure: bool = Field(default=False, description="Set true to simulate Order API outage/timeout")

class ToolExecutionLog(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    output: Any

class ChatResponse(BaseModel):
    session_id: str
    customer_id: str
    llm_response: str = Field(description="Natural language answer from the conversational agent")
    authoritative_operational_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Unmodified ground-truth data returned by operational tools (Req 2A)"
    )
    automation_workflow_triggered: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured complaint classification and ticket details when workflow runs (Req 2D)"
    )
    observability_trace: List[ToolExecutionLog] = []