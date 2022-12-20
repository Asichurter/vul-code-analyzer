from tree_sitter import Language, Parser

# Language.build_library(
#   # Store the library in the `build` directory
#   'build/my-languages.so',
#   # Include one or more languages
#   [
#     '../tools/tree-sitter-cpp',
# '../tools/tree-sitter-c',
#   ]
# )

C_CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')

parser = Parser()
parser.set_language(C_CPP_LANGUAGE)

# code_file_path = '/data1/zhijietang/temp/cparser_test_file.c'
code_file_path = '/data1/zhijietang/temp/treesitter_test_file_1.cpp'

def read_code_to_parse():
    with open(code_file_path, 'r', encoding='utf-8') as f:
        return f.read().encode('utf-8')

tree = parser.parse(read_code_to_parse())
root = tree.root_node




















code = """static uid_t get_caller_uid(GDBusConnection *connection, GDBusMethodInvocation *invocation, const char *caller)
 {
     GError *error = NULL;
     guint caller_uid;

     GDBusProxy * proxy = g_dbus_proxy_new_sync(connection,
                                      G_DBUS_PROXY_FLAGS_NONE,
                                      NULL,
                                      "org.freedesktop.DBus",
                                      "/org/freedesktop/DBus",
                                      "org.freedesktop.DBus",
                                      NULL,
                                      &error);

     GVariant *result = g_dbus_proxy_call_sync(proxy,
                                      "GetConnectionUnixUser",
                                      g_variant_new ("(s)", caller),
                                      G_DBUS_CALL_FLAGS_NONE,
                                      -1,
                                      NULL,
                                      &error);

     if (result == NULL)
     {
         /* we failed to get the uid, so return (uid_t) -1 to indicate the error
          */
         if (error)
         {
             g_dbus_method_invocation_return_dbus_error(invocation,
                                       "org.freedesktop.problems.InvalidUser",
                                       error->message);
             g_error_free(error);
         }
         else
         {
             g_dbus_method_invocation_return_dbus_error(invocation,
                                       "org.freedesktop.problems.InvalidUser",
                                       _("Unknown error"));
         }
         return (uid_t) -1;
     }

     g_variant_get(result, "(u)", &caller_uid);
     g_variant_unref(result);

     log_info("Caller uid: %i", caller_uid);
     return caller_uid;
 }"""