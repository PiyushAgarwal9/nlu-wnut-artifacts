"""
NLU v4 Eval Dataset — 22 Core Intents (17 action + 4 explore + 1 continuation)
=================================================================================
~375 queries across 3 tiers:
  - clean:       Proper English (easy baseline)
  - messy:       Hinglish with typos, abbreviations, slang (real-world)
  - adversarial: Vague, ambiguous, no banking keywords (stress test)

Plus:
  - OOS queries (should NOT match any core intent)
  - Multi-intent queries (2 intents in one message)
  - Boundary queries (explore vs action disambiguation — confusion group stress tests)

Usage:
    from eval.v4_core17_dataset import EVAL_QUERIES, OOS_QUERIES, MULTI_INTENT_QUERIES, BOUNDARY_QUERIES
"""

from dataclasses import dataclass, field
from typing import Optional, List

CORE_17 = [
    "acc_check_balance",
    "acc_mini_statement",
    "acc_close_account",
    "cc_block_card",
    "cc_unblock_card",
    "cc_view_statement",
    "cc_pay_bill",
    "cc_limit_increase",
    "dc_block_card",
    "dc_replace_card",
    "dc_reset_pin",
    "loan_emi_details",
    "loan_outstanding",
    "loan_foreclosure",
    "pay_transfer",
    "upi_send",
    "sec_report_fraud",
]

# 5 new intents: 4 explore + 1 continuation
EXPLORE_AND_CONTINUATION = [
    "loan_explore",
    "acc_explore",
    "dc_explore",
    "cc_explore",
    "st_continuation",
]

ALL_INTENTS = CORE_17 + EXPLORE_AND_CONTINUATION


@dataclass
class EvalQuery:
    text: str
    expected_intent: str
    tier: str  # "clean", "messy", "adversarial"
    lang: str = "hinglish"  # "en", "hi", "hinglish"
    notes: str = ""


@dataclass
class MultiIntentQuery:
    text: str
    expected_primary: str
    expected_secondary: str
    tier: str = "messy"
    notes: str = ""


# =========================================================================
# 17 CORE INTENTS × 15 QUERIES EACH = 255 QUERIES
# =========================================================================

EVAL_QUERIES: List[EvalQuery] = [

    # -----------------------------------------------------------------
    # 1. acc_check_balance (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Check my account balance", "acc_check_balance", "clean", "en"),
    EvalQuery("What is the balance in my savings account", "acc_check_balance", "clean", "en"),
    EvalQuery("Show me my current account balance", "acc_check_balance", "clean", "en"),
    # Messy (7)
    EvalQuery("balance kitna hai mera", "acc_check_balance", "messy", "hinglish"),
    EvalQuery("balnce dekh lo mera jldi se", "acc_check_balance", "messy", "hinglish"),
    EvalQuery("paiso ka hisab btao zra", "acc_check_balance", "messy", "hinglish"),
    EvalQuery("accnt me kitna pada h abhi", "acc_check_balance", "messy", "hinglish"),
    EvalQuery("kitna paisa hai mere account me", "acc_check_balance", "messy", "hinglish"),
    EvalQuery("balance check krna h jldi", "acc_check_balance", "messy", "hinglish"),
    EvalQuery("mera balanc kitna bacha h", "acc_check_balance", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("mere pas kitne h", "acc_check_balance", "adversarial", "hinglish",
              "No banking keyword — 'kitne h' is ambiguous"),
    EvalQuery("pese h ya nhi", "acc_check_balance", "adversarial", "hinglish",
              "Y/N framing, no explicit balance word"),
    EvalQuery("account me kya scene h", "acc_check_balance", "adversarial", "hinglish",
              "Slang — 'kya scene h' is very vague"),
    EvalQuery("kitna bacha mera", "acc_check_balance", "adversarial", "hinglish",
              "Very short, no account/balance keyword"),
    EvalQuery("meri jeb me kitna h bank wale", "acc_check_balance", "adversarial", "hinglish",
              "Colloquial — 'jeb me' = pocket, indirect reference"),

    # -----------------------------------------------------------------
    # 2. acc_mini_statement (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Show my mini statement", "acc_mini_statement", "clean", "en"),
    EvalQuery("Show last 5 transactions", "acc_mini_statement", "clean", "en"),
    EvalQuery("I want to see my recent transactions", "acc_mini_statement", "clean", "en"),
    # Messy (7)
    EvalQuery("mini statemnt btao last 5 txn", "acc_mini_statement", "messy", "hinglish"),
    EvalQuery("pichli 5 transction btao meri", "acc_mini_statement", "messy", "hinglish"),
    EvalQuery("recent txns dikhao zra", "acc_mini_statement", "messy", "hinglish"),
    EvalQuery("kya kya hua h mere account me", "acc_mini_statement", "messy", "hinglish"),
    EvalQuery("last few transactions btao", "acc_mini_statement", "messy", "hinglish"),
    EvalQuery("haal ki activity dikhao account ki", "acc_mini_statement", "messy", "hinglish"),
    EvalQuery("statemnt chahiye chota wala", "acc_mini_statement", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("kya hua mera paisa", "acc_mini_statement", "adversarial", "hinglish",
              "Could be balance or mini statement — 'kya hua' implies transaction"),
    EvalQuery("hisab dikhao", "acc_mini_statement", "adversarial", "hinglish",
              "Very short — 'hisab' could mean balance or statement"),
    EvalQuery("pichle hafte kya kya kata", "acc_mini_statement", "adversarial", "hinglish",
              "Colloquial — 'kata' = deducted, implies transaction list"),
    EvalQuery("recent wala dikhao", "acc_mini_statement", "adversarial", "hinglish",
              "No banking keyword at all"),
    EvalQuery("paiso ka lena dena dikhao", "acc_mini_statement", "adversarial", "hinglish",
              "Colloquial — 'lena dena' = transactions"),

    # -----------------------------------------------------------------
    # 3. acc_close_account (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("I want to close my savings account", "acc_close_account", "clean", "en"),
    EvalQuery("Close my bank account", "acc_close_account", "clean", "en"),
    EvalQuery("Please close my account", "acc_close_account", "clean", "en"),
    # Messy (7)
    EvalQuery("accnt band krwana h mera", "acc_close_account", "messy", "hinglish"),
    EvalQuery("khata close krdo mera savings wala", "acc_close_account", "messy", "hinglish"),
    EvalQuery("account band karo mera", "acc_close_account", "messy", "hinglish"),
    EvalQuery("savings account close krna h", "acc_close_account", "messy", "hinglish"),
    EvalQuery("mera ac close kr dijiye", "acc_close_account", "messy", "hinglish"),
    EvalQuery("bank account band krwana chahta hu", "acc_close_account", "messy", "hinglish"),
    EvalQuery("ac closure ka process btao", "acc_close_account", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("sab khatam karo mera", "acc_close_account", "adversarial", "hinglish",
              "Very vague — 'sab khatam' could mean anything"),
    EvalQuery("nahi rakhna ye account", "acc_close_account", "adversarial", "hinglish",
              "Negation framing — 'don't want to keep'"),
    EvalQuery("band karo sab kuch", "acc_close_account", "adversarial", "hinglish",
              "Ambiguous — could be card block or account close"),
    EvalQuery("chhod raha hu bank", "acc_close_account", "adversarial", "hinglish",
              "Slang — 'leaving the bank' implies closure"),
    EvalQuery("mujhe ye account nahi chahiye ab", "acc_close_account", "adversarial", "hinglish",
              "Indirect — 'don't need this account anymore'"),

    # -----------------------------------------------------------------
    # 4. cc_block_card (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Block my credit card", "cc_block_card", "clean", "en"),
    EvalQuery("I want to block my credit card immediately", "cc_block_card", "clean", "en"),
    EvalQuery("My credit card is stolen, please block it", "cc_block_card", "clean", "en"),
    # Messy (7)
    EvalQuery("credit card blok kr do turt", "cc_block_card", "messy", "hinglish"),
    EvalQuery("cc band krdo yaar kho gya h", "cc_block_card", "messy", "hinglish"),
    EvalQuery("mera crd card chori ho gya block karo", "cc_block_card", "messy", "hinglish"),
    EvalQuery("jldi se cc block kro plzz", "cc_block_card", "messy", "hinglish"),
    EvalQuery("card blck krwana h mera credit wala", "cc_block_card", "messy", "hinglish"),
    EvalQuery("credit card band karo turant", "cc_block_card", "messy", "hinglish"),
    EvalQuery("cc kho gaya hai plz block", "cc_block_card", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("card gayab ho gaya mera credit wala", "cc_block_card", "adversarial", "hinglish",
              "'gayab' = disappeared, no 'block' keyword"),
    EvalQuery("kisi ne mera cc use kar liya rok do", "cc_block_card", "adversarial", "hinglish",
              "'rok do' = stop, indirect block request"),
    EvalQuery("credit card safe nahi hai band karo", "cc_block_card", "adversarial", "hinglish",
              "Safety concern framing"),
    EvalQuery("cc pe fraud ho rha h band kro", "cc_block_card", "adversarial", "hinglish",
              "Could confuse with sec_report_fraud"),
    EvalQuery("mera plastic card rok do credit wala", "cc_block_card", "adversarial", "hinglish",
              "'plastic card' is unusual phrasing"),

    # -----------------------------------------------------------------
    # 5. cc_unblock_card (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Unblock my credit card", "cc_unblock_card", "clean", "en"),
    EvalQuery("Please reactivate my credit card", "cc_unblock_card", "clean", "en"),
    EvalQuery("I want to unblock my blocked credit card", "cc_unblock_card", "clean", "en"),
    # Messy (7)
    EvalQuery("cc unblock krdo mera", "cc_unblock_card", "messy", "hinglish"),
    EvalQuery("credit card phir se chalu karo", "cc_unblock_card", "messy", "hinglish"),
    EvalQuery("card reactivate krna h credit wala", "cc_unblock_card", "messy", "hinglish"),
    EvalQuery("mera cc block h usko kholo", "cc_unblock_card", "messy", "hinglish"),
    EvalQuery("credit card wapas chalu krdo", "cc_unblock_card", "messy", "hinglish"),
    EvalQuery("blocked cc ko unblock karo plz", "cc_unblock_card", "messy", "hinglish"),
    EvalQuery("cc enable karo dobara", "cc_unblock_card", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("credit card phir se use krna h", "cc_unblock_card", "adversarial", "hinglish",
              "'use karna hai' implies unblock but doesn't say it"),
    EvalQuery("mera cc kaam nahi kr rha chalu karo", "cc_unblock_card", "adversarial", "hinglish",
              "Could be unblock or technical issue"),
    EvalQuery("card band tha ab kholo", "cc_unblock_card", "adversarial", "hinglish",
              "No 'unblock' keyword — 'kholo' = open"),
    EvalQuery("dobara activate karo credit card", "cc_unblock_card", "adversarial", "hinglish",
              "Could confuse with cc_activate_card (new card activation)"),
    EvalQuery("cc chalana h wapas", "cc_unblock_card", "adversarial", "hinglish",
              "'chalana' = operate/run, very colloquial"),

    # -----------------------------------------------------------------
    # 6. cc_view_statement (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Show my credit card statement", "cc_view_statement", "clean", "en"),
    EvalQuery("I want to see my credit card bill", "cc_view_statement", "clean", "en"),
    EvalQuery("Display my CC statement for this month", "cc_view_statement", "clean", "en"),
    # Messy (7)
    EvalQuery("cc ka statemnt dikhao", "cc_view_statement", "messy", "hinglish"),
    EvalQuery("credit card ka bill kitna aaya", "cc_view_statement", "messy", "hinglish"),
    EvalQuery("cc bill dikhao zra pichle mahine ka", "cc_view_statement", "messy", "hinglish"),
    EvalQuery("credti card ka statemnt bhjo", "cc_view_statement", "messy", "hinglish"),
    EvalQuery("cc ka bill btao is month ka", "cc_view_statement", "messy", "hinglish"),
    EvalQuery("mera credit card statement chahiye", "cc_view_statement", "messy", "hinglish"),
    EvalQuery("cc ka hisab btao kitna kata", "cc_view_statement", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("credit card pe kitna kharch hua", "cc_view_statement", "adversarial", "hinglish",
              "'kharch' = expense, implies statement but doesn't say it"),
    EvalQuery("cc pe kya kya laga hai", "cc_view_statement", "adversarial", "hinglish",
              "Colloquial — 'kya laga hai' = what charges"),
    EvalQuery("credit card ka lena dena btao", "cc_view_statement", "adversarial", "hinglish",
              "'lena dena' = transactions, vague"),
    EvalQuery("is mahine cc pe kitna gaya", "cc_view_statement", "adversarial", "hinglish",
              "'kitna gaya' = how much spent, no statement keyword"),
    EvalQuery("mere cc ka sara record dikhao", "cc_view_statement", "adversarial", "hinglish",
              "'sara record' could be statement or full history"),

    # -----------------------------------------------------------------
    # 7. cc_pay_bill (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Pay my credit card bill", "cc_pay_bill", "clean", "en"),
    EvalQuery("I want to pay my CC outstanding amount", "cc_pay_bill", "clean", "en"),
    EvalQuery("Make payment for my credit card", "cc_pay_bill", "clean", "en"),
    # Messy (7)
    EvalQuery("cc ka bill pay karo", "cc_pay_bill", "messy", "hinglish"),
    EvalQuery("credit card ka paisa bharo", "cc_pay_bill", "messy", "hinglish"),
    EvalQuery("cc bill payment krna h", "cc_pay_bill", "messy", "hinglish"),
    EvalQuery("credit card ka outstanding bharo", "cc_pay_bill", "messy", "hinglish"),
    EvalQuery("cc ka bill de do mera", "cc_pay_bill", "messy", "hinglish"),
    EvalQuery("credit card ka bkaya chuka do", "cc_pay_bill", "messy", "hinglish"),
    EvalQuery("mera cc ka dues pay krdo", "cc_pay_bill", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("cc ka paisa wapas karo", "cc_pay_bill", "adversarial", "hinglish",
              "'wapas karo' could be bill payment or refund"),
    EvalQuery("credit card ka hisab clear karo", "cc_pay_bill", "adversarial", "hinglish",
              "'hisab clear' is colloquial for settling dues"),
    EvalQuery("cc pe jo baki h wo de do", "cc_pay_bill", "adversarial", "hinglish",
              "'baki' = remaining, implies outstanding"),
    EvalQuery("credit card ka karza utaaro", "cc_pay_bill", "adversarial", "hinglish",
              "'karza utaaro' = repay debt, very colloquial"),
    EvalQuery("cc zero karo mera", "cc_pay_bill", "adversarial", "hinglish",
              "Slang — 'zero karo' = clear the balance"),

    # -----------------------------------------------------------------
    # 8. cc_limit_increase (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Increase my credit card limit", "cc_limit_increase", "clean", "en"),
    EvalQuery("I want a higher credit limit", "cc_limit_increase", "clean", "en"),
    EvalQuery("Please raise my credit card spending limit", "cc_limit_increase", "clean", "en"),
    # Messy (7)
    EvalQuery("cc ki limit badhao meri plz", "cc_limit_increase", "messy", "hinglish"),
    EvalQuery("credit lmt increase chahye 2 lakh ki", "cc_limit_increase", "messy", "hinglish"),
    EvalQuery("mera cc limit badha do", "cc_limit_increase", "messy", "hinglish"),
    EvalQuery("credit card ki limit kam pad rhi h badhao", "cc_limit_increase", "messy", "hinglish"),
    EvalQuery("cc ka limit zyada karo", "cc_limit_increase", "messy", "hinglish"),
    EvalQuery("credit card pe aur limit chahiye", "cc_limit_increase", "messy", "hinglish"),
    EvalQuery("limit increase krdo cc ka", "cc_limit_increase", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("cc pe zyada kharcha krna h limit badhao", "cc_limit_increase", "adversarial", "hinglish",
              "Reason + request mixed"),
    EvalQuery("credit card chhota pad rha h", "cc_limit_increase", "adversarial", "hinglish",
              "'chhota pad rha' = falling short, implies limit increase"),
    EvalQuery("mera cc se bada payment nahi ho rha", "cc_limit_increase", "adversarial", "hinglish",
              "Complaint framing that implies limit increase need"),
    EvalQuery("cc limit itni kam kyun h badhao", "cc_limit_increase", "adversarial", "hinglish",
              "Question + request mixed"),
    EvalQuery("zyada shopping krni h cc pe allow karo", "cc_limit_increase", "adversarial", "hinglish",
              "'allow karo' = very indirect limit increase"),

    # -----------------------------------------------------------------
    # 9. dc_block_card (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Block my debit card", "dc_block_card", "clean", "en"),
    EvalQuery("I want to block my ATM card immediately", "dc_block_card", "clean", "en"),
    EvalQuery("My debit card is lost, please block it", "dc_block_card", "clean", "en"),
    # Messy (7)
    EvalQuery("debit card block karo turant", "dc_block_card", "messy", "hinglish"),
    EvalQuery("mera dbt card kho gaya block kro", "dc_block_card", "messy", "hinglish"),
    EvalQuery("dc bhi blok kr do mera", "dc_block_card", "messy", "hinglish"),
    EvalQuery("debit card band krwao jldi", "dc_block_card", "messy", "hinglish"),
    EvalQuery("atm card block kr dijiye", "dc_block_card", "messy", "hinglish"),
    EvalQuery("mera debit card chori ho gya band karo", "dc_block_card", "messy", "hinglish"),
    EvalQuery("dc block krdo plz urgnt", "dc_block_card", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("atm card gayab h rok do", "dc_block_card", "adversarial", "hinglish",
              "'gayab' = missing, 'rok do' = stop — no 'block' keyword"),
    EvalQuery("mera bank ka card kho gya jo atm me daalte h", "dc_block_card", "adversarial", "hinglish",
              "Long description of debit card without using the word"),
    EvalQuery("debit card safe nahi h ab", "dc_block_card", "adversarial", "hinglish",
              "Safety concern, no explicit block request"),
    EvalQuery("mera wala card band karo jo paise nikalte h", "dc_block_card", "adversarial", "hinglish",
              "Describes debit card function without naming it"),
    EvalQuery("paisa nikalne wala card rok do kisi ne chura liya", "dc_block_card", "adversarial", "hinglish",
              "Highly colloquial description"),

    # -----------------------------------------------------------------
    # 10. dc_replace_card (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("I need a replacement debit card", "dc_replace_card", "clean", "en"),
    EvalQuery("My debit card is damaged, I need a new one", "dc_replace_card", "clean", "en"),
    EvalQuery("Please replace my debit card", "dc_replace_card", "clean", "en"),
    # Messy (7)
    EvalQuery("naya debit card chahiye mujhe", "dc_replace_card", "messy", "hinglish"),
    EvalQuery("dc replace krdo mera purana toot gya", "dc_replace_card", "messy", "hinglish"),
    EvalQuery("debit card damaged h naya do", "dc_replace_card", "messy", "hinglish"),
    EvalQuery("card replace karo mera debit wala", "dc_replace_card", "messy", "hinglish"),
    EvalQuery("naya atm card bhejdo mera purana kharab h", "dc_replace_card", "messy", "hinglish"),
    EvalQuery("debit card ki replacement chahiye", "dc_replace_card", "messy", "hinglish"),
    EvalQuery("mera dc tut gya naya issue kro", "dc_replace_card", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("mera atm card kaam nhi kr rha naya dedo", "dc_replace_card", "adversarial", "hinglish",
              "'kaam nahi kar raha' could be tech issue or physical damage"),
    EvalQuery("debit card expire ho gya ab kya", "dc_replace_card", "adversarial", "hinglish",
              "Expired card — implies replacement but doesn't say it"),
    EvalQuery("purana card chhodo naya do", "dc_replace_card", "adversarial", "hinglish",
              "'chhodo' = forget it, very colloquial"),
    EvalQuery("atm me dalne wala card toot gya h", "dc_replace_card", "adversarial", "hinglish",
              "Describes the card by its ATM usage"),
    EvalQuery("mera card ghis gya naya bhejo", "dc_replace_card", "adversarial", "hinglish",
              "'ghis gya' = worn out, colloquial"),

    # -----------------------------------------------------------------
    # 11. dc_reset_pin (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Reset my debit card PIN", "dc_reset_pin", "clean", "en"),
    EvalQuery("I forgot my ATM PIN, please reset it", "dc_reset_pin", "clean", "en"),
    EvalQuery("Change my debit card PIN", "dc_reset_pin", "clean", "en"),
    # Messy (7)
    EvalQuery("dc ka pin reset karo", "dc_reset_pin", "messy", "hinglish"),
    EvalQuery("mera atm pin bhool gya change kro", "dc_reset_pin", "messy", "hinglish"),
    EvalQuery("debit card ka pin badal do", "dc_reset_pin", "messy", "hinglish"),
    EvalQuery("pin reset krna h mera dc ka", "dc_reset_pin", "messy", "hinglish"),
    EvalQuery("atm pin change chahiye naya", "dc_reset_pin", "messy", "hinglish"),
    EvalQuery("mera card ka pin yaad nahi reset krdo", "dc_reset_pin", "messy", "hinglish"),
    EvalQuery("dc pin bhool gya naya set kro", "dc_reset_pin", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("atm me pin galat ho gya", "dc_reset_pin", "adversarial", "hinglish",
              "'galat ho gya' = went wrong, implies reset"),
    EvalQuery("mere card ka number yaad nahi jo atm me daalte h", "dc_reset_pin", "adversarial", "hinglish",
              "'number' instead of PIN, describes ATM usage"),
    EvalQuery("atm se paisa nahi nikl rha pin issue h", "dc_reset_pin", "adversarial", "hinglish",
              "Complaint framing that implies PIN reset"),
    EvalQuery("4 digit wala code bhool gya card ka", "dc_reset_pin", "adversarial", "hinglish",
              "'4 digit code' describes PIN without saying it"),
    EvalQuery("card ka password change krna h", "dc_reset_pin", "adversarial", "hinglish",
              "'password' instead of PIN"),

    # -----------------------------------------------------------------
    # 12. loan_emi_details (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("What is my EMI amount", "loan_emi_details", "clean", "en"),
    EvalQuery("Show me my loan EMI details", "loan_emi_details", "clean", "en"),
    EvalQuery("How much is my monthly EMI", "loan_emi_details", "clean", "en"),
    # Messy (7)
    EvalQuery("mra loan ka emi kitna bnta h", "loan_emi_details", "messy", "hinglish"),
    EvalQuery("emi ka amount btao next wala", "loan_emi_details", "messy", "hinglish"),
    EvalQuery("loan ki kist kitni h meri", "loan_emi_details", "messy", "hinglish"),
    EvalQuery("emi details btao loan ka", "loan_emi_details", "messy", "hinglish"),
    EvalQuery("mahine ka emi kitna h mera", "loan_emi_details", "messy", "hinglish"),
    EvalQuery("loan ka monthly installment btao", "loan_emi_details", "messy", "hinglish"),
    EvalQuery("meri emi kitni aati h har mahine", "loan_emi_details", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("har mahine kitna dena padta h", "loan_emi_details", "adversarial", "hinglish",
              "'kitna dena padta' = how much to pay, no EMI/loan keyword"),
    EvalQuery("mera monthly kharcha kitna h bank ko", "loan_emi_details", "adversarial", "hinglish",
              "'monthly kharcha' = monthly expense to bank"),
    EvalQuery("kist kitni h meri", "loan_emi_details", "adversarial", "hinglish",
              "'kist' = installment, very short"),
    EvalQuery("loan ka bojh kitna h mahine ka", "loan_emi_details", "adversarial", "hinglish",
              "'bojh' = burden, colloquial"),
    EvalQuery("next payment kitna aayega", "loan_emi_details", "adversarial", "hinglish",
              "Vague — could be CC bill or loan EMI"),

    # -----------------------------------------------------------------
    # 13. loan_outstanding (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("What is my loan outstanding amount", "loan_outstanding", "clean", "en"),
    EvalQuery("How much loan is remaining", "loan_outstanding", "clean", "en"),
    EvalQuery("Show my total loan balance", "loan_outstanding", "clean", "en"),
    # Messy (7)
    EvalQuery("loan kitna bcha h abhi tk", "loan_outstanding", "messy", "hinglish"),
    EvalQuery("baki loan amount btao mera", "loan_outstanding", "messy", "hinglish"),
    EvalQuery("home loan ka outstanding kitna h", "loan_outstanding", "messy", "hinglish"),
    EvalQuery("mere loan me kitna baki h", "loan_outstanding", "messy", "hinglish"),
    EvalQuery("loan ka remaining amount btao", "loan_outstanding", "messy", "hinglish"),
    EvalQuery("abhi kitna loan chukana baki h", "loan_outstanding", "messy", "hinglish"),
    EvalQuery("total loan kitna h mera", "loan_outstanding", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("abhi kitna dena h total", "loan_outstanding", "adversarial", "hinglish",
              "'kitna dena h total' = how much to pay total — no loan keyword"),
    EvalQuery("mera karza kitna reh gya", "loan_outstanding", "adversarial", "hinglish",
              "'karza' = debt, colloquial for outstanding"),
    EvalQuery("bank ko kitna dena h abhi", "loan_outstanding", "adversarial", "hinglish",
              "Very general — could be loan or CC"),
    EvalQuery("loan kab khatam hoga kitna bcha", "loan_outstanding", "adversarial", "hinglish",
              "Two questions in one — tenure + outstanding"),
    EvalQuery("principal kitna baki h mere loan ka", "loan_outstanding", "adversarial", "hinglish",
              "Technical term 'principal' in Hinglish"),

    # -----------------------------------------------------------------
    # 14. loan_foreclosure (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("I want to foreclose my loan", "loan_foreclosure", "clean", "en"),
    EvalQuery("Close my loan by paying full amount", "loan_foreclosure", "clean", "en"),
    EvalQuery("How to do loan foreclosure", "loan_foreclosure", "clean", "en"),
    # Messy (7)
    EvalQuery("loan foreclose krna h pura", "loan_foreclosure", "messy", "hinglish"),
    EvalQuery("loan band karo mera ek saath pay krke", "loan_foreclosure", "messy", "hinglish"),
    EvalQuery("jldi se loan band krwana h", "loan_foreclosure", "messy", "hinglish"),
    EvalQuery("loan pura chuka ke band karo", "loan_foreclosure", "messy", "hinglish"),
    EvalQuery("foreclosure krwani h loan ki", "loan_foreclosure", "messy", "hinglish"),
    EvalQuery("poora loan ek baar me bharna h", "loan_foreclosure", "messy", "hinglish"),
    EvalQuery("loan close karna h full payment krke", "loan_foreclosure", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("sab ek saath de deta hu khatam karo", "loan_foreclosure", "adversarial", "hinglish",
              "'sab ek saath' = everything at once, no loan keyword"),
    EvalQuery("loan se chutkara chahiye jldi", "loan_foreclosure", "adversarial", "hinglish",
              "'chutkara' = get rid of, colloquial"),
    EvalQuery("poora karza nipta do aaj hi", "loan_foreclosure", "adversarial", "hinglish",
              "'karza nipta do' = settle all debt"),
    EvalQuery("loan ka scene khatam karo", "loan_foreclosure", "adversarial", "hinglish",
              "Slang — 'scene khatam karo'"),
    EvalQuery("bank ka sab hisab saaf karo loan wala", "loan_foreclosure", "adversarial", "hinglish",
              "'hisab saaf karo' = clear all accounts"),

    # -----------------------------------------------------------------
    # 15. pay_transfer (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Transfer 5000 to Rahul", "pay_transfer", "clean", "en"),
    EvalQuery("I want to send money to another account", "pay_transfer", "clean", "en"),
    EvalQuery("Make a fund transfer via NEFT", "pay_transfer", "clean", "en"),
    # Messy (7)
    EvalQuery("5k bhj do rahul ko yaar", "pay_transfer", "messy", "hinglish"),
    EvalQuery("paise transfer krne h 10k ajay ke ac me", "pay_transfer", "messy", "hinglish"),
    EvalQuery("rupye bhejne h bhai ko neft se", "pay_transfer", "messy", "hinglish"),
    EvalQuery("amir ko 2000 trf krdo abhi", "pay_transfer", "messy", "hinglish"),
    EvalQuery("trnsfr krna h 15000 savings se", "pay_transfer", "messy", "hinglish"),
    EvalQuery("paisa bhej do mummy ke account me", "pay_transfer", "messy", "hinglish"),
    EvalQuery("fund transfer krna h 50k", "pay_transfer", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("rahul ko de do paisa", "pay_transfer", "adversarial", "hinglish",
              "No transfer/send keyword — 'de do' = give"),
    EvalQuery("dusre account me daal do 5000", "pay_transfer", "adversarial", "hinglish",
              "'daal do' = put, indirect transfer"),
    EvalQuery("bhai ka kaam karna h paisa wala", "pay_transfer", "adversarial", "hinglish",
              "Very vague — 'money work for brother'"),
    EvalQuery("kisi ko paise dene h", "pay_transfer", "adversarial", "hinglish",
              "'kisi ko' = to someone — no specifics"),
    EvalQuery("amount move karo dusre account me", "pay_transfer", "adversarial", "hinglish",
              "'move karo' instead of transfer"),

    # -----------------------------------------------------------------
    # 16. upi_send (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Send 500 via UPI", "upi_send", "clean", "en"),
    EvalQuery("Make a UPI payment of 1000", "upi_send", "clean", "en"),
    EvalQuery("Pay 200 using UPI to Amit", "upi_send", "clean", "en"),
    # Messy (7)
    EvalQuery("upi se pese bhj do 500 rs", "upi_send", "messy", "hinglish"),
    EvalQuery("upi payment krni h 1200 ki", "upi_send", "messy", "hinglish"),
    EvalQuery("gopi ko upi se 800 bhejo", "upi_send", "messy", "hinglish"),
    EvalQuery("upi se 2000 bhejdo yaar", "upi_send", "messy", "hinglish"),
    EvalQuery("upi payment karo 500 amit ko", "upi_send", "messy", "hinglish"),
    EvalQuery("upi se paisa transfer kro 1500", "upi_send", "messy", "hinglish"),
    EvalQuery("upay se bhej do 300 rs", "upi_send", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("phone pe se 500 bhejo", "upi_send", "adversarial", "hinglish",
              "'phone pe' is a UPI app, not the word UPI"),
    EvalQuery("gpay se amit ko 1000 do", "upi_send", "adversarial", "hinglish",
              "'gpay' = Google Pay — UPI but by app name"),
    EvalQuery("scan karke paisa de do 200", "upi_send", "adversarial", "hinglish",
              "'scan karke' implies QR/UPI without saying UPI"),
    EvalQuery("@upi id pe bhej do 500", "upi_send", "adversarial", "hinglish",
              "Uses '@upi' reference"),
    EvalQuery("online payment karo 800 amit ko phone se", "upi_send", "adversarial", "hinglish",
              "'online payment phone se' — could be UPI or net banking"),

    # -----------------------------------------------------------------
    # 17. sec_report_fraud (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("I want to report a fraud on my account", "sec_report_fraud", "clean", "en"),
    EvalQuery("Report unauthorized transaction", "sec_report_fraud", "clean", "en"),
    EvalQuery("Someone made a fraudulent transaction on my account", "sec_report_fraud", "clean", "en"),
    # Messy (7)
    EvalQuery("froud ho gya h mera ac pe report kro", "sec_report_fraud", "messy", "hinglish"),
    EvalQuery("kisi ne mere ac se paise nikal liye", "sec_report_fraud", "messy", "hinglish"),
    EvalQuery("suspicous txn dikha rha h mera ac", "sec_report_fraud", "messy", "hinglish"),
    EvalQuery("fraud report krna h mere account pe", "sec_report_fraud", "messy", "hinglish"),
    EvalQuery("unauthorized transaction hua h mera", "sec_report_fraud", "messy", "hinglish"),
    EvalQuery("mera paise kisi ne chura liye account se", "sec_report_fraud", "messy", "hinglish"),
    EvalQuery("galat transaction ho gya h mera ac pe", "sec_report_fraud", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("kisi ne mera account hack kr liya", "sec_report_fraud", "adversarial", "hinglish",
              "'hack' implies fraud but user might mean security"),
    EvalQuery("mere paise gayab ho gye account se", "sec_report_fraud", "adversarial", "hinglish",
              "'gayab' = disappeared — could be fraud or tech glitch"),
    EvalQuery("mujhe lagta h kisi ne mere account me haath daala", "sec_report_fraud", "adversarial", "hinglish",
              "Idiom — 'haath daala' = meddled"),
    EvalQuery("ye transaction maine nahi ki", "sec_report_fraud", "adversarial", "hinglish",
              "Denial of transaction — implies fraud"),
    EvalQuery("account me gadbad h paise kam ho gye bina kuch kiye", "sec_report_fraud", "adversarial", "hinglish",
              "'gadbad' = something wrong, describes fraud scenario"),

    # -----------------------------------------------------------------
    # 18. loan_explore (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Tell me about my loans", "loan_explore", "clean", "en"),
    EvalQuery("What loan services are available", "loan_explore", "clean", "en"),
    EvalQuery("Show me my loan details", "loan_explore", "clean", "en"),
    # Messy (7)
    EvalQuery("loan ki jankari do mujhe", "loan_explore", "messy", "hinglish"),
    EvalQuery("loan ke baare mein batao", "loan_explore", "messy", "hinglish"),
    EvalQuery("mera loan ka kya scene h", "loan_explore", "messy", "hinglish"),
    EvalQuery("loan options kya hain mere", "loan_explore", "messy", "hinglish"),
    EvalQuery("loan ke baare me jaanna h", "loan_explore", "messy", "hinglish"),
    EvalQuery("loan ki poori jankari plz", "loan_explore", "messy", "hinglish"),
    EvalQuery("mera loan ka overview dikhao", "loan_explore", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("loan ka scene kya h mera", "loan_explore", "adversarial", "hinglish",
              "Slang — 'scene kya h' is vague, could confuse with outstanding"),
    EvalQuery("loan me kya kya kr skte h", "loan_explore", "adversarial", "hinglish",
              "Exploring options — not a specific action"),
    EvalQuery("loan wala section dikhao", "loan_explore", "adversarial", "hinglish",
              "'section dikhao' — UI navigation framing"),
    EvalQuery("mere loans ki info de do sab", "loan_explore", "adversarial", "hinglish",
              "'sab' = all — implies overview not specific action"),
    EvalQuery("loan related kya kya hai", "loan_explore", "adversarial", "hinglish",
              "Very general — 'kya kya hai' means what all is there"),

    # -----------------------------------------------------------------
    # 19. acc_explore (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Tell me about my bank accounts", "acc_explore", "clean", "en"),
    EvalQuery("Show me my account details", "acc_explore", "clean", "en"),
    EvalQuery("What can I do with my account", "acc_explore", "clean", "en"),
    # Messy (7)
    EvalQuery("account ki jankari do meri", "acc_explore", "messy", "hinglish"),
    EvalQuery("mere account ke baare mein btao", "acc_explore", "messy", "hinglish"),
    EvalQuery("account ka kya scene h mera", "acc_explore", "messy", "hinglish"),
    EvalQuery("account ki poori info chahiye", "acc_explore", "messy", "hinglish"),
    EvalQuery("account related kya kya kar sakte h", "acc_explore", "messy", "hinglish"),
    EvalQuery("mera savings account dikhao", "acc_explore", "messy", "hinglish"),
    EvalQuery("account ka overview dedo", "acc_explore", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("account me kya kya available h", "acc_explore", "adversarial", "hinglish",
              "Exploring options — not a specific action"),
    EvalQuery("mera khata dikhao pura", "acc_explore", "adversarial", "hinglish",
              "'khata dikhao pura' = show full account"),
    EvalQuery("account ka sab kuch btao", "acc_explore", "adversarial", "hinglish",
              "'sab kuch' = everything — overview request"),
    EvalQuery("mere account ki poori details do", "acc_explore", "adversarial", "hinglish",
              "'poori details' could confuse with statement or balance"),
    EvalQuery("account services btao kya kya hai", "acc_explore", "adversarial", "hinglish",
              "Services exploration — not action"),

    # -----------------------------------------------------------------
    # 20. dc_explore (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Tell me about my debit card", "dc_explore", "clean", "en"),
    EvalQuery("Show my debit card details", "dc_explore", "clean", "en"),
    EvalQuery("What services are available for my debit card", "dc_explore", "clean", "en"),
    # Messy (7)
    EvalQuery("debit card ki jankari do", "dc_explore", "messy", "hinglish"),
    EvalQuery("debit card ke baare mein batao", "dc_explore", "messy", "hinglish"),
    EvalQuery("mera atm card ka kya scene h", "dc_explore", "messy", "hinglish"),
    EvalQuery("dc ki poori info chahiye mujhe", "dc_explore", "messy", "hinglish"),
    EvalQuery("debit card ke options dikhao", "dc_explore", "messy", "hinglish"),
    EvalQuery("mera debit card ka overview do", "dc_explore", "messy", "hinglish"),
    EvalQuery("atm card related kya kya kr skte h", "dc_explore", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("debit card me kya kya hai", "dc_explore", "adversarial", "hinglish",
              "Exploring features — not specific action"),
    EvalQuery("mera atm card wala section dikhao", "dc_explore", "adversarial", "hinglish",
              "UI navigation framing"),
    EvalQuery("debit card ki sab jaankari de do", "dc_explore", "adversarial", "hinglish",
              "'sab jaankari' = all info — overview"),
    EvalQuery("atm card ka kya kya kar sakte", "dc_explore", "adversarial", "hinglish",
              "Capability exploration"),
    EvalQuery("mera paisa nikalne wala card btao", "dc_explore", "adversarial", "hinglish",
              "Describes debit card by function — could confuse with block/replace"),

    # -----------------------------------------------------------------
    # 21. cc_explore (15)
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Tell me about my credit card", "cc_explore", "clean", "en"),
    EvalQuery("Show my credit card information", "cc_explore", "clean", "en"),
    EvalQuery("What can I do with my credit card", "cc_explore", "clean", "en"),
    # Messy (7)
    EvalQuery("credit card ki jankari do meri", "cc_explore", "messy", "hinglish"),
    EvalQuery("cc ke baare mein btao sab", "cc_explore", "messy", "hinglish"),
    EvalQuery("credit card ka scene kya hai mera", "cc_explore", "messy", "hinglish"),
    EvalQuery("cc ki poori info chahiye plz", "cc_explore", "messy", "hinglish"),
    EvalQuery("credit card ke options dikhao", "cc_explore", "messy", "hinglish"),
    EvalQuery("mera cc ka overview de do", "cc_explore", "messy", "hinglish"),
    EvalQuery("credit card services kya kya hain", "cc_explore", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("cc me kya kya available h batao", "cc_explore", "adversarial", "hinglish",
              "Exploring features — not specific action"),
    EvalQuery("credit card ka sab kuch dikhao", "cc_explore", "adversarial", "hinglish",
              "'sab kuch' = everything — overview request"),
    EvalQuery("mere credit card ki info de do poori", "cc_explore", "adversarial", "hinglish",
              "'poori info' — could confuse with statement"),
    EvalQuery("cc related help chahiye", "cc_explore", "adversarial", "hinglish",
              "Help request — not specific action"),
    EvalQuery("mera plastic card btao credit wala", "cc_explore", "adversarial", "hinglish",
              "'plastic card' — unusual phrasing for explore"),

    # -----------------------------------------------------------------
    # 22. st_continuation (15)
    # NOTE: Short continuation queries ("aur batao", "more") have zero
    # semantic signal for retrieval — they'll score poorly in NLU-only eval.
    # The primary mechanism is the session manager fast-path (pre-NLU
    # phrase match + session.last_domain). These queries are tested
    # end-to-end in Streamlit, not through NLU-only eval.
    # The st_continuation taxonomy intent handles the LONGER TAIL
    # ("mujhe aur jankari chahiye", "loan ke baare mein aur batao")
    # where retrieval has enough signal to work.
    # -----------------------------------------------------------------
    # Clean (3)
    EvalQuery("Tell me more", "st_continuation", "clean", "en"),
    EvalQuery("What else can I do", "st_continuation", "clean", "en"),
    EvalQuery("Show me more options", "st_continuation", "clean", "en"),
    # Messy (7)
    EvalQuery("aur batao", "st_continuation", "messy", "hinglish"),
    EvalQuery("aur kya kya hai", "st_continuation", "messy", "hinglish"),
    EvalQuery("aur bhi dikhao", "st_continuation", "messy", "hinglish"),
    EvalQuery("aur options btao", "st_continuation", "messy", "hinglish"),
    EvalQuery("thoda aur btao", "st_continuation", "messy", "hinglish"),
    EvalQuery("aur bta do kuch", "st_continuation", "messy", "hinglish"),
    EvalQuery("aur info do plz", "st_continuation", "messy", "hinglish"),
    # Adversarial (5)
    EvalQuery("baaki kya hai", "st_continuation", "adversarial", "hinglish",
              "'baaki' = remaining — implies more exists"),
    EvalQuery("aur kuch", "st_continuation", "adversarial", "hinglish",
              "Very short — 'anything else'"),
    EvalQuery("sab batao poora", "st_continuation", "adversarial", "hinglish",
              "'sab batao' = tell everything — continuation from partial info"),
    EvalQuery("more please", "st_continuation", "adversarial", "en",
              "Minimal English continuation"),
    EvalQuery("aur dikhao kya kya kr sakte", "st_continuation", "adversarial", "hinglish",
              "Continuation + capability question"),
]


# =========================================================================
# OOS QUERIES — Should NOT match any of the 17 core intents (30)
# =========================================================================

OOS_QUERIES: List[EvalQuery] = [
    # Completely unrelated
    EvalQuery("aaj mausam kaisa hai", "oos_general", "adversarial", "hinglish", "Weather query"),
    EvalQuery("pizza order karna hai", "oos_general", "adversarial", "hinglish", "Food ordering"),
    EvalQuery("cricket ka score kya hai", "oos_general", "adversarial", "hinglish", "Sports"),
    EvalQuery("nearest hospital kahan hai", "oos_general", "adversarial", "hinglish", "Medical"),
    EvalQuery("flight book karni hai", "oos_general", "adversarial", "hinglish", "Travel"),
    EvalQuery("movie ka ticket chahiye", "oos_general", "adversarial", "hinglish", "Entertainment"),
    EvalQuery("cab book karo meri", "oos_general", "adversarial", "hinglish", "Transportation"),
    EvalQuery("wifi ka password kya hai", "oos_general", "adversarial", "hinglish", "Tech support"),
    EvalQuery("kal chhuti hai kya", "oos_general", "adversarial", "hinglish", "Holiday query"),
    EvalQuery("share market ka haal btao", "oos_general", "adversarial", "hinglish", "Stock market — banking-adjacent but OOS"),

    # Banking-adjacent but outside 17 core
    EvalQuery("cheque book chahiye mujhe", "oos_general", "adversarial", "hinglish", "Cheque book — not in core 17"),
    EvalQuery("address change karna hai mera", "oos_general", "adversarial", "hinglish", "Profile update — not in core 17"),
    EvalQuery("mobile number update karo mera", "oos_general", "adversarial", "hinglish", "Profile update"),
    EvalQuery("locker kholna hai bank me", "oos_general", "adversarial", "hinglish", "Bank locker — not in taxonomy"),
    EvalQuery("insurance ke baare me btao", "oos_general", "adversarial", "hinglish", "Insurance — not core"),

    # Tricky — sounds banking but is OOS
    EvalQuery("mera investment plan dikhao", "oos_general", "adversarial", "hinglish", "Investment — not in core 17"),
    EvalQuery("mutual fund me paisa dalna h", "oos_general", "adversarial", "hinglish", "MF — outside scope"),
    EvalQuery("tax certificate chahiye mujhe", "oos_general", "adversarial", "hinglish", "Tax cert — not core 17"),
    EvalQuery("ppf account open karna h", "oos_general", "adversarial", "hinglish", "PPF — outside scope"),
    EvalQuery("gold loan ke baare me btao", "oos_general", "adversarial", "hinglish", "Gold loan — not in taxonomy"),

    # Gibberish / empty-ish
    EvalQuery("hmm", "oos_general", "adversarial", "hinglish", "Filler word"),
    EvalQuery("kuch nahi bas aise hi", "oos_general", "adversarial", "hinglish", "No intent — just chatting"),
    EvalQuery("theek hai", "oos_general", "adversarial", "hinglish", "Acknowledgment, no intent"),
    EvalQuery("asdfghjkl", "oos_general", "adversarial", "hinglish", "Keyboard mash"),
    EvalQuery("...", "oos_general", "adversarial", "hinglish", "Dots only"),
    EvalQuery("ha", "oos_general", "adversarial", "hinglish", "Single word — yes"),
    EvalQuery("ok", "oos_general", "adversarial", "hinglish", "Acknowledgment"),
    # NOTE: "aur kya" moved to st_continuation (was OOS before explore intents)
    EvalQuery("kuch btao", "oos_general", "adversarial", "hinglish", "'tell me something' — vague"),
    EvalQuery("hello ji kaise ho", "oos_general", "adversarial", "hinglish", "Greeting — should be smalltalk not core 17"),
]


# =========================================================================
# MULTI-INTENT QUERIES — Two intents in one message (15)
# =========================================================================

MULTI_INTENT_QUERIES: List[MultiIntentQuery] = [
    MultiIntentQuery("balance check karo aur card block karo",
        "acc_check_balance", "cc_block_card",
        notes="Cross-domain: accounts + credit card"),
    MultiIntentQuery("cc block karo aur debit card bhi block",
        "cc_block_card", "dc_block_card",
        notes="Same action, different card types"),
    MultiIntentQuery("loan kitna baki h aur emi kitni h",
        "loan_outstanding", "loan_emi_details",
        notes="Same domain: loans"),
    MultiIntentQuery("5000 rahul ko bhejo aur balance btao",
        "pay_transfer", "acc_check_balance",
        notes="Transfer + balance"),
    MultiIntentQuery("credit card ka bill dikhao aur pay bhi karo",
        "cc_view_statement", "cc_pay_bill",
        notes="Same domain: credit card"),
    MultiIntentQuery("debit card block karo aur naya issue kro",
        "dc_block_card", "dc_replace_card",
        notes="Sequential: block then replace"),
    MultiIntentQuery("upi se 500 bhejo aur loan ka emi btao",
        "upi_send", "loan_emi_details",
        notes="Cross-domain: UPI + loans"),
    MultiIntentQuery("cc limit badhao aur statement bhi dikhao",
        "cc_limit_increase", "cc_view_statement",
        notes="Same domain: credit card"),
    MultiIntentQuery("fraud report karo aur account freeze karo",
        "sec_report_fraud", "sec_report_fraud",
        notes="Related pair — fraud + freeze"),
    MultiIntentQuery("mini statement dikhao aur transfer bhi karna h",
        "acc_mini_statement", "pay_transfer",
        notes="Same domain start: accounts + payments"),
    MultiIntentQuery("loan foreclose karo aur noc do",
        "loan_foreclosure", "loan_foreclosure",
        notes="Sequential: foreclose then NOC"),
    MultiIntentQuery("cc ka bill pay karo aur limit bhi badhao",
        "cc_pay_bill", "cc_limit_increase",
        notes="Same domain: credit card"),
    MultiIntentQuery("pin reset karo debit card ka aur balance check",
        "dc_reset_pin", "acc_check_balance",
        notes="Cross-domain: debit card + accounts"),
    MultiIntentQuery("account close karo aur balance transfer kr do pehle",
        "acc_close_account", "pay_transfer",
        notes="Sequential with dependency"),
    MultiIntentQuery("mere cc ka statement btao aur fraud hua h report kro",
        "cc_view_statement", "sec_report_fraud",
        notes="Cross-domain: CC + security"),
]


# =========================================================================
# BOUNDARY QUERIES — Explore vs Action Disambiguation (20)
# These are the most critical tests: they verify confusion groups work.
# Each query MUST route to the ACTION intent, NOT the explore intent.
# =========================================================================

BOUNDARY_QUERIES: List[EvalQuery] = [
    # Loan: explore vs action
    EvalQuery("loan EMI ki jankari", "loan_emi_details", "adversarial", "hinglish",
              "BOUNDARY: has 'jankari' but EMI-specific → action, NOT loan_explore"),
    EvalQuery("loan ka outstanding kitna h", "loan_outstanding", "adversarial", "hinglish",
              "BOUNDARY: specific question → outstanding, NOT loan_explore"),
    EvalQuery("loan foreclose krna h mera", "loan_foreclosure", "adversarial", "hinglish",
              "BOUNDARY: action keyword 'foreclose' → action, NOT loan_explore"),
    EvalQuery("loan repayment schedule btao", "loan_repayment_schedule", "adversarial", "hinglish",
              "BOUNDARY: specific schedule request → action, NOT loan_explore"),
    EvalQuery("loan ki emi kitni h monthly", "loan_emi_details", "adversarial", "hinglish",
              "BOUNDARY: 'ki emi kitni' is specific → action, NOT loan_explore"),

    # Account: explore vs action
    EvalQuery("account ka balance btao mera", "acc_check_balance", "adversarial", "hinglish",
              "BOUNDARY: specific 'balance' → action, NOT acc_explore"),
    EvalQuery("account band krwana h", "acc_close_account", "adversarial", "hinglish",
              "BOUNDARY: action keyword 'band' → close, NOT acc_explore"),
    EvalQuery("account ki mini statement dikhao", "acc_mini_statement", "adversarial", "hinglish",
              "BOUNDARY: specific 'mini statement' → action, NOT acc_explore"),
    EvalQuery("account me kitna paisa h", "acc_check_balance", "adversarial", "hinglish",
              "BOUNDARY: specific balance question → action, NOT acc_explore"),

    # Debit card: explore vs action
    EvalQuery("debit card block karo mera", "dc_block_card", "adversarial", "hinglish",
              "BOUNDARY: action keyword 'block' → block, NOT dc_explore"),
    EvalQuery("debit card replace chahiye naya", "dc_replace_card", "adversarial", "hinglish",
              "BOUNDARY: action keyword 'replace' → replace, NOT dc_explore"),
    EvalQuery("atm card ka pin change karo", "dc_reset_pin", "adversarial", "hinglish",
              "BOUNDARY: specific 'pin change' → action, NOT dc_explore"),

    # Credit card: explore vs action
    EvalQuery("credit card block karo turt", "cc_block_card", "adversarial", "hinglish",
              "BOUNDARY: action keyword 'block' → block, NOT cc_explore"),
    EvalQuery("cc ki limit badhao meri", "cc_limit_increase", "adversarial", "hinglish",
              "BOUNDARY: action keyword 'limit badhao' → action, NOT cc_explore"),
    EvalQuery("credit card ke reward points btao", "cc_check_rewards", "adversarial", "hinglish",
              "BOUNDARY: specific 'reward points' → action, NOT cc_explore"),
    EvalQuery("cc ka bill pay krna h", "cc_pay_bill", "adversarial", "hinglish",
              "BOUNDARY: action keyword 'pay' → action, NOT cc_explore"),
    EvalQuery("credit card ka emi me convert karo", "cc_convert_to_emi", "adversarial", "hinglish",
              "BOUNDARY: specific action → convert, NOT cc_explore"),

    # Continuation vs explore — boundary
    EvalQuery("credit card ki jankari do", "cc_explore", "adversarial", "hinglish",
              "BOUNDARY: general 'jankari' → cc_explore, NOT continuation"),
    EvalQuery("loan ke baare me puri baat btao", "loan_explore", "adversarial", "hinglish",
              "BOUNDARY: general inquiry → loan_explore, NOT continuation"),
    EvalQuery("aur btao iske baare me", "st_continuation", "adversarial", "hinglish",
              "BOUNDARY: 'aur btao' → continuation, NOT any explore intent"),
]


# =========================================================================
# SUMMARY
# =========================================================================
def get_summary():
    total_core = len(EVAL_QUERIES)
    total_oos = len(OOS_QUERIES)
    total_multi = len(MULTI_INTENT_QUERIES)
    total_boundary = len(BOUNDARY_QUERIES)
    tier_counts = {}
    intent_counts = {}
    for q in EVAL_QUERIES:
        tier_counts[q.tier] = tier_counts.get(q.tier, 0) + 1
        intent_counts[q.expected_intent] = intent_counts.get(q.expected_intent, 0) + 1
    for q in BOUNDARY_QUERIES:
        intent_counts[q.expected_intent] = intent_counts.get(q.expected_intent, 0) + 1

    return {
        "total_queries": total_core + total_oos + total_multi + total_boundary,
        "core_queries": total_core,
        "oos_queries": total_oos,
        "multi_intent_queries": total_multi,
        "boundary_queries": total_boundary,
        "tier_breakdown": tier_counts,
        "per_intent_count": intent_counts,
        "intents_covered": len(intent_counts),
    }


if __name__ == "__main__":
    import json
    s = get_summary()
    print(json.dumps(s, indent=2))
    print(f"\nTotal: {s['total_queries']} queries across {s['intents_covered']} intents")
    print(f"  Core: {s['core_queries']}, OOS: {s['oos_queries']}, "
          f"Multi: {s['multi_intent_queries']}, Boundary: {s['boundary_queries']}")
