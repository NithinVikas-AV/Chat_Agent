from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool

@tool()
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return f"Hello, {name}!"

class ReverseStringArgs(BaseModel):
    text: str = Field(description="Text to be reversed")

@tool(args_schema=ReverseStringArgs)
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]

class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First string")
    b: str = Field(description="Second string")

@tool(args_schema=ConcatenateStringsArgs)
def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    print("a", a)
    print("b", b)
    return a + b