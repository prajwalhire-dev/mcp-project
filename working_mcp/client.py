import os
import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
import nest_asyncio
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

class SQLAgentClient:
    """Orchestrates a multi-step AI pipeline to answer questions using a database."""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio: Optional[Any] = None
        self.write: Optional[Any] = None
        print("SQL Agent Client initialized.")

    async def connect(self, server_script_path: str = "server.py"):
        """Connects to the MCP server."""
        print(f"Connecting to MCP server: {server_script_path}...")
        server_params = StdioServerParameters(command="python", args=[server_script_path])
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            await self.session.initialize()

            tools_result = await self.session.list_tools()
            print("\nSuccessfully connected. Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}")
        except Exception as e:
            print(f"Failed to connect: {e}")
            raise

    async def ask(self, question: str) -> str:
        """Processes a question through the full 4-tool pipeline."""
        if not self.session:
            return "Error: Not connected to the server."

        print(f"\nProcessing question: \"{question}\"")
        try:
            # Step 1: Extract Entities with NER
            print("Step 1: Calling ner_generator_dynamic...")
            ner_result = await self.session.call_tool("ner_generator_dynamic", {"question": question})
            # The tool now returns a dictionary directly.
            ner_dict = ner_result.content[0].data
            print(f" -> NER Result: {ner_dict}")

            # Step 2: Create SQL Query
            print("Step 2: Calling create_sql...")
            # Pass the dictionary to the next tool.
            sql_result = await self.session.call_tool("create_sql", {"question": question, "ner_dict": ner_dict})
            # The tool returns a JSON string, which is correct.
            sql_json_str = sql_result.content[0].text
            print(f" -> SQL Created: {sql_json_str}")

            # Step 3: Run Query on Database
            print("Step 3: Calling run_sqlite_query...")
            db_result = await self.session.call_tool("run_sqlite_query", {"sql_json": sql_json_str})
            # The tool now returns a dictionary directly.
            db_dict = db_result.content[0].data
            print(f" -> Database query executed.")

            # Step 4: Generate Final Answer
            print("Step 4: Calling generate_final_answer...")
            # Pass the dictionary to the final tool.
            final_answer = await self.session.call_tool("generate_final_answer", {"question": question, "query_result_dict": db_dict})
            print(" -> Final answer generated.")
            
            return final_answer.content[0].text

        except Exception as e:
            return f"A critical error occurred in the pipeline: {e}"

    async def cleanup(self):
        """Closes the connection."""
        print("\nCleaning up and closing connection...")
        await self.exit_stack.aclose()
        print("Connection closed.")

async def main():
    """Main function to run the client."""
    client = SQLAgentClient()
    try:
        await client.connect()
        
        # --- Example Question ---
        question = "What is the maximum base MSRP in King county?"
        response = await client.ask(question)
        
        print("\n" + "="*50)
        print(f"Question: {question}")
        print(f"Final Answer: {response}")
        print("="*50)

    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())