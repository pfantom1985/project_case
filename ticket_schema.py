from typing import Literal
from pydantic import BaseModel, Field


Category = Literal[
    "billing",
    "technical_issue",
    "account_access",
    "feature_request",
    "general_question",
    "complaint",
    "other",
]

Priority = Literal[
    "low",
    "medium",
    "high",
    "urgent",
]

ReplyLanguage = Literal[
    "ru",
    "en",
]


class TicketStructuredOutput(BaseModel):
    category: Category = Field(
        ...,
        description="Main support ticket category."
    )
    priority: Priority = Field(
        ...,
        description="Priority level based on urgency and impact."
    )
    needs_human: bool = Field(
        ...,
        description="Whether the case requires handoff to a human agent."
    )
    reply_language: ReplyLanguage = Field(
        ...,
        description="Language that should be used in the draft reply."
    )
    draft_reply: str = Field(
        ...,
        min_length=1,
        description="Short draft reply to the customer in the detected language."
    )