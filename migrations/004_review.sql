ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS review jsonb;
ALTER TABLE agent_turns ADD COLUMN IF NOT EXISTS review_status text;
