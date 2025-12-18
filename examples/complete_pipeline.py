"""
Complete Content Creation Pipeline Example
From: The Architect's Playbook

This example demonstrates all five pillars working together:
- MCP for tool integration
- Visual AI patterns
- LangGraph for orchestration
- Flight Deck for governance
- Type safety and testing patterns
"""

import asyncio
import json
import logging
from typing import TypedDict, List, Annotated
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Import our pillar implementations
import sys
sys.path.append('..')
from pillar_4_flight_deck.governor import AgentExecutionGovernor, CostTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION (Pillar 5: Type Safety)
# ============================================================================

class ContentState(TypedDict):
    """Typed state for content creation pipeline"""
    topic: str
    target_audience: str
    content_type: str
    
    # Research phase
    research_summary: str
    sources: List[dict]
    
    # Writing phase
    draft_content: str
    word_count: int
    
    # Editing phase
    quality_score: float
    feedback: List[str]
    revision_count: int
    
    # Output
    final_content: str
    total_cost: float


# ============================================================================
# AGENT IMPLEMENTATIONS (Pillar 3: LangGraph)
# ============================================================================

class ResearchAgent:
    """Specialist in finding and validating sources"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0.3)
    
    def __call__(self, state: ContentState) -> dict:
        logger.info(f"Researching: {state['topic']}")
        
        response = self.llm.invoke(
            f"Research the topic: {state['topic']}\n"
            f"Target audience: {state['target_audience']}\n"
            f"Provide 3-5 key findings with sources. Return JSON format."
        )
        
        return {
            'research_summary': response.content,
            'sources': []  # Would parse from response
        }


class WritingAgent:
    """Specialist in creating engaging content"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0.7)
    
    def __call__(self, state: ContentState) -> dict:
        logger.info(f"Writing {state['content_type']}")
        
        feedback_context = ""
        if state.get('revision_count', 0) > 0 and state.get('feedback'):
            feedback_context = f"\n\nAddress this feedback:\n" + \
                              "\n".join(f"- {f}" for f in state['feedback'])
        
        response = self.llm.invoke(
            f"Write a {state['content_type']} about: {state['topic']}\n"
            f"Audience: {state['target_audience']}\n"
            f"Research: {state['research_summary']}"
            f"{feedback_context}\n\n"
            f"Write 800-1200 words."
        )
        
        draft = response.content
        return {
            'draft_content': draft,
            'word_count': len(draft.split()),
            'revision_count': state.get('revision_count', 0) + 1
        }


class EditorAgent:
    """Specialist in reviewing and improving content"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0.2)
    
    def __call__(self, state: ContentState) -> dict:
        logger.info("Editing content")
        
        response = self.llm.invoke(
            f"Review this {state['content_type']}:\n\n"
            f"{state['draft_content']}\n\n"
            f"Rate quality 1-10 and provide specific feedback. "
            f"Return JSON: {{\"score\": N, \"feedback\": [...]}}"
        )
        
        try:
            review = json.loads(response.content)
        except:
            review = {"score": 7.0, "feedback": ["Unable to parse review"]}
        
        return {
            'quality_score': review.get('score', 7.0),
            'feedback': review.get('feedback', [])
        }


class PublisherAgent:
    """Specialist in formatting and publishing"""
    
    def __call__(self, state: ContentState) -> dict:
        logger.info("Publishing content")
        
        return {
            'final_content': state['draft_content'],
            'published_url': f"https://example.com/blog/{state['topic'].replace(' ', '-').lower()}"
        }


# ============================================================================
# WORKFLOW CONSTRUCTION
# ============================================================================

def route_after_edit(state: ContentState) -> str:
    """Route based on quality score"""
    if state.get('revision_count', 0) >= 3:
        logger.info("Max revisions reached, publishing")
        return 'publish'
    
    score = state.get('quality_score', 0)
    if score >= 8.0:
        return 'publish'
    elif score >= 6.0:
        return 'revise'
    else:
        return 'research'


def create_content_workflow():
    """Build the multi-agent content creation workflow"""
    workflow = StateGraph(ContentState)
    
    # Add agent nodes
    workflow.add_node('research', ResearchAgent())
    workflow.add_node('write', WritingAgent())
    workflow.add_node('edit', EditorAgent())
    workflow.add_node('publish', PublisherAgent())
    
    # Define edges
    workflow.set_entry_point('research')
    workflow.add_edge('research', 'write')
    workflow.add_edge('write', 'edit')
    
    workflow.add_conditional_edges('edit', route_after_edit, {
        'publish': 'publish',
        'revise': 'write',
        'research': 'research'
    })
    
    workflow.add_edge('publish', END)
    
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION WITH GOVERNANCE (Pillar 4: Flight Deck)
# ============================================================================

async def main():
    """Run the complete pipeline with governance"""
    
    # Initialize governance
    governor = AgentExecutionGovernor(
        max_cost_per_execution=2.0,
        max_execution_time=300,
        max_retries=3
    )
    
    # Create workflow
    workflow = create_content_workflow()
    
    # Initial state
    initial_state = {
        'topic': 'Production AI Agent Architecture',
        'target_audience': 'Software Engineers',
        'content_type': 'technical blog post',
        'revision_count': 0,
        'sources': [],
        'feedback': []
    }
    
    logger.info("Starting content creation pipeline...")
    
    # Execute with governance
    def run_workflow():
        return workflow.invoke(initial_state)
    
    result = governor.execute(run_workflow)
    
    if result.success:
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Duration: {result.duration:.2f}s")
        logger.info(f"Cost: ${result.cost:.4f}")
        print("\n" + "="*50)
        print("FINAL CONTENT:")
        print("="*50)
        print(result.result.get('final_content', 'No content')[:500] + "...")
    else:
        logger.error(f"Pipeline failed: {result.error}")


if __name__ == '__main__':
    asyncio.run(main())
