import sqlite3

def check_db():
    print("--- 🔍 INSPECTING MEMORY DATABASE ---")
    try:
        conn = sqlite3.connect("chatbot.db")
        cursor = conn.cursor()
        
        print("\n📖 SESSION SUMMARIES (Long Term Memory):")
        cursor.execute("SELECT id, date_str, topic_summary FROM session_summaries")
        rows = cursor.fetchall()
        if not rows:
            print("   [EMPTY] No summaries found.")
        else:
            for row in rows:
                print(f"   ID {row[0]} ({row[1]}): {row[2]}")

        print("\n💬 RECENT CHAT HISTORY:")
        cursor.execute("SELECT sender, message FROM chat_history ORDER BY id DESC LIMIT 5")
        rows = cursor.fetchall()
        if not rows:
            print("   [EMPTY] No recent messages.")
        else:
            for row in rows:
                print(f"   {row[0]}: {row[1]}")
                
        conn.close()
    except Exception as e:
        print(f"❌ Error reading database: {e}")
    print("\n---------------------------------------")

if __name__ == "__main__":
    check_db()