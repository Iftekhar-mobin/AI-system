## thanks
* thank
  - utter_noworries
  - utter_anything_else

## bye
* bye
  - utter_bye

## greet
* greet
  - action_greet_user
  - utter_greet

## happy path
* greet
  - utter_greet
  - action_greet_user
* ask_howdoing
  - utter_thumbsup

## sad path 1
* greet
  - utter_greet
  - action_greet_user
* canthelp
  - utter_ask_whatspossible
  - utter_faq
  - utter_what_help
  - utter_ask_whatisamie
  - action_default_fallback
* affirm
  - action_default_ask_affirmation
* good_work
  - utter_noworries
  - utter_anything_else
* bye
    - utter_bye

## say goodbye
* bye
  - utter_bye

## new to amie
* greet
  - utter_greet
  - action_greet_user
* ask_whatspossible
  - utter_getstarted
  - utter_ask_whatspossible
  - utter_faq

## chitchat
* greet
  - utter_greet
  - action_greet_user
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
  - action_chitchat
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
  - action_chitchat
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
  - action_chitchat
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
  - action_chitchat
* canthelp
  - utter_not_sure
  - utter_ask_whatspossible
* deny
  - action_default_fallback
  - utter_ask_whatspossible
  - utter_faq
  - action_default_ask_affirmation

* thank
  - utter_noworries
  - utter_thumbsup
  - utter_anything_else
  - utter_praise
* bye
  - utter_bye

## anything else? - yes
* greet
  - utter_greet
  - action_greet_user
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
  - action_chitchat
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
  - action_chitchat
* thank
  - utter_anything_else
* affirm
  - utter_what_help
* bye
  - utter_bye

## anything else? - no
* greet
  - utter_greet
  - action_greet_user
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
  - action_chitchat
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
  - action_chitchat
* thank
  - utter_anything_else
* deny
  - utter_thumbsup
  - action_default_ask_affirmation

## anything else?
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
    - action_chitchat
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
    - action_chitchat
* ask_whatspossible
    - utter_ask_whatspossible
    - utter_tutorial
* ask_question_in_forum
    - utter_ask_question_in_forum

## greet and chitchat
* greet
  - utter_greet
  - action_greet_user
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
    - action_chitchat
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
    - action_chitchat
* handleinsult
    - utter_not_sure
    - utter_ask_whatisamie
* bye
  - utter_bye

## Customer support needed
* greet
  - utter_greet
  - action_greet_user
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_howbuilt
    - action_chitchat
* contact_staffs
    - utter_contact_staffs
    - utter_need_customer_support
* react_negative
    - utter_negative
    - utter_explain

## faq needed
* faq
    - utter_faq
    - utter_ask_whatspossible

## chitchat to good work
* ask_builder OR ask_howdoing OR ask_whoisit OR ask_whatisamie OR ask_isbot OR ask_howold OR ask_languagesbot OR ask_wherefrom OR ask_howbuilt OR ask_whatspossible
    - action_chitchat
* good_work
    - utter_good_work

## canthelp and deny
* how_to_get_started
    - utter_getstarted
* good_work
    - utter_good_work
* praise
    - utter_praise
* canthelp
    - utter_canthelp
* deny
    - utter_nohelp

## handleinsult and denial
* deny
    - utter_nohelp
* when_to_use
    - utter_when_to_use
* ask_whatspossible
    - utter_ask_whatspossible
    - utter_faq
* handleinsult
    - utter_handleinsult

## canthelp insulted
* canthelp
    - utter_canthelp
* handleinsult
    - utter_handleinsult

## new to amie only
* how_to_get_started
    - utter_getstarted

## good_work only
* good_work
    - utter_good_work

## handleinsult only
* handleinsult
    - utter_handleinsult

## praise only
* praise
    - utter_praise

## when_to_use only
* when_to_use
    - utter_when_to_use

## greet only
* greet
  - utter_greet
  - action_greet_user

## deny followed by praise
* deny
    - utter_nohelp
* how_to_get_started
    - utter_getstarted
    - utter_faq
* good_work
    - utter_good_work
* praise
    - utter_praise

## greet and start
* greet
  - utter_greet
  - action_greet_user
* when_to_use
    - utter_when_to_use
* how_to_get_started
    - utter_getstarted
* bye
  - utter_bye

## greet and continue
* greet
  - utter_greet
  - action_greet_user
* how_to_get_started
    - utter_getstarted
* when_to_use
    - utter_when_to_use
* ask_whatspossible
    - utter_ask_whatspossible
    - utter_faq
* ask_languagesbot
    - utter_ask_languagesbot
* canthelp
    - utter_canthelp
* thank
    - utter_noworries
    - utter_anything_else
* bye
    - utter_bye
