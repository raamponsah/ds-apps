import ldap
import traceback

def test_ldap_bind():
    server_uri = "ldap://svr-cbgad-w1p.cbg.com"
    bind_dn = "CN=bind_user,CN=Users,DC=cbg,DC=com"
    bind_password = "bind_password"

    try:
        conn = ldap.initialize(server_uri)
        conn.set_option(ldap.OPT_REFERRALS, 0)
        conn.set_option(ldap.OPT_DEBUG_LEVEL, 255)
        conn.simple_bind_s(bind_dn, bind_password)
        print("Bind successful!")
        conn.unbind_s()
    except ldap.INVALID_CREDENTIALS:
        print("Bind failed: Invalid credentials")
        traceback.print_exc()
    except ldap.LDAPError as e:
        print(f"LDAP error: {e}")
        traceback.print_exc()

test_ldap_bind()