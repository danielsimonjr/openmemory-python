"""
Simple example of using OpenMemory Python
"""
import asyncio
from openmemory import MemorySystem


async def main():
    # Initialize memory system
    print("Initializing OpenMemory...")
    memory = MemorySystem(db_path="example.db")

    # Add some memories
    print("\nAdding memories...")

    result1 = await memory.add_memory(
        content="The user prefers dark mode in their IDE and terminal",
        user_id="user123",
        tags=["preference", "ui"]
    )
    print(f"Added memory: {result1['id']} (sector: {result1['primary_sector']})")

    result2 = await memory.add_memory(
        content="Yesterday I went to the coffee shop and had a great latte",
        user_id="user123",
        tags=["experience", "food"]
    )
    print(f"Added memory: {result2['id']} (sector: {result2['primary_sector']})")

    result3 = await memory.add_memory(
        content="To install Python packages, use pip install package-name",
        user_id="user123",
        tags=["knowledge", "programming"]
    )
    print(f"Added memory: {result3['id']} (sector: {result3['primary_sector']})")

    result4 = await memory.add_memory(
        content="I'm really excited about this new AI project!",
        user_id="user123",
        tags=["emotion", "project"]
    )
    print(f"Added memory: {result4['id']} (sector: {result4['primary_sector']})")

    # Query memories
    print("\nQuerying memories...")

    print("\n1. Query: 'What are the user's preferences?'")
    results = await memory.query(
        query="What are the user's preferences?",
        user_id="user123",
        k=3
    )
    for i, mem in enumerate(results, 1):
        print(f"   {i}. [score: {mem.score:.3f}] {mem.content[:70]}...")
        print(f"      Sector: {mem.primary_sector}, Salience: {mem.salience:.3f}")

    print("\n2. Query: 'Tell me about coffee'")
    results = await memory.query(
        query="Tell me about coffee",
        user_id="user123",
        k=3
    )
    for i, mem in enumerate(results, 1):
        print(f"   {i}. [score: {mem.score:.3f}] {mem.content[:70]}...")
        print(f"      Sector: {mem.primary_sector}, Salience: {mem.salience:.3f}")

    print("\n3. Query: 'How do I install software?'")
    results = await memory.query(
        query="How do I install software?",
        user_id="user123",
        k=3
    )
    for i, mem in enumerate(results, 1):
        print(f"   {i}. [score: {mem.score:.3f}] {mem.content[:70]}...")
        print(f"      Sector: {mem.primary_sector}, Salience: {mem.salience:.3f}")

    # Reinforce a memory
    print(f"\nReinforcing memory about preferences...")
    await memory.reinforce_memory(result1['id'], boost=0.2)

    # Query again to see reinforcement effect
    print("\nQuerying again after reinforcement...")
    results = await memory.query(
        query="user settings",
        user_id="user123",
        k=2
    )
    for i, mem in enumerate(results, 1):
        print(f"   {i}. [score: {mem.score:.3f}, salience: {mem.salience:.3f}]")
        print(f"      {mem.content[:70]}...")

    # Cleanup
    memory.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
